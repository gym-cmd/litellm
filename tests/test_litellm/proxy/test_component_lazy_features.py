"""Tests for ``restrict_lazy_features_to_component``.

The componentized deployments (``gateway/main.py``, ``backend/main.py``) trim
``app.router.routes`` once at startup, but ``LazyFeatureMiddleware`` registers
optional routers on demand at request time. Without the restrict-to-component
hook, a single request to a management-only path like ``/guardrails`` on a
gateway pod would import and mount that router after the trim has already
finished, leaking ~20 admin routes onto the data plane.

These tests exercise the helper against an isolated ``FastAPI`` instance
(rather than the shared proxy app) so no test runs in this file mutate state
that other tests depend on.
"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from litellm.proxy._lazy_features import (
    LAZY_FEATURES,
    LazyFeature,
    LazyFeatureMiddleware,
    restrict_lazy_features_to_component,
)


def _make_app_with_lazy_middleware(features=LAZY_FEATURES) -> FastAPI:
    """Build a FastAPI app with the lazy middleware attached (default
    ``LAZY_FEATURES`` unless the caller passes a smaller list)."""
    app = FastAPI()
    app.add_middleware(LazyFeatureMiddleware, fastapi_app=app, features=features)
    return app


def _make_test_features() -> tuple[LazyFeature, ...]:
    """Two synthetic lazy features so tests don't depend on the real
    LAZY_FEATURES tuple (which evolves and would force test churn)."""

    def _register_admin(app: FastAPI, _module) -> None:
        @app.get("/admin/things")
        def _things():
            return {}

        @app.get("/admin/things/{thing_id}")
        def _thing(thing_id: str):
            return {}

    def _register_partial(app: FastAPI, _module) -> None:
        @app.get("/data/items")
        def _items():
            return {}

        @app.get("/admin/items/{item_id}/audit")
        def _audit(item_id: str):
            return {}

    return (
        LazyFeature(
            name="admin_only",
            module_path="json",  # any always-importable stdlib module
            path_prefixes=("/admin/things",),
            register_fn=_register_admin,
        ),
        LazyFeature(
            name="partial_overlap",
            module_path="csv",
            path_prefixes=("/data/items", "/admin/items"),
            register_fn=_register_partial,
        ),
    )


async def _hit(mw: LazyFeatureMiddleware, path: str) -> None:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_msg):
        pass

    await mw(
        {
            "type": "http",
            "path": path,
            "method": "GET",
            "headers": [],
            "query_string": b"",
            "raw_path": path.encode(),
        },
        receive,
        send,
    )


def _route_paths(app: FastAPI) -> set[str]:
    return {r.path for r in app.router.routes if isinstance(r, APIRoute)}


@pytest.mark.asyncio
async def test_admin_only_feature_is_dropped_on_data_plane(monkeypatch):
    """A lazy feature whose every prefix sits outside the data-plane allowlist
    must never be considered by the middleware -- a request to that prefix
    must NOT import the module."""
    test_features = _make_test_features()
    monkeypatch.setattr(
        "litellm.proxy._lazy_features.LAZY_FEATURES", test_features, raising=True
    )

    app = _make_app_with_lazy_middleware(features=test_features)

    def trim(target_app: FastAPI) -> None:
        target_app.router.routes = [
            r
            for r in target_app.router.routes
            if not isinstance(r, APIRoute) or r.path.startswith("/data/")
        ]

    restrict_lazy_features_to_component(
        app,
        allowed_path_prefixes=("/data/",),
        allowed_exact_paths=frozenset(),
        re_trim=trim,
    )

    spec = next(s for s in app.user_middleware if s.cls is LazyFeatureMiddleware)
    mw = LazyFeatureMiddleware(app, **spec.kwargs)

    await _hit(mw, "/admin/things")

    assert "/admin/things" not in _route_paths(app), (
        "admin-only lazy feature was loaded despite no overlap with allowlist; "
        "the heavy module should never be imported on the wrong component"
    )


@pytest.mark.asyncio
async def test_partial_overlap_feature_keeps_only_allowed_routes(monkeypatch):
    """A lazy feature with both data-plane and admin prefixes loads when one
    overlap matches, but the admin routes it registers get trimmed by the
    wrapped register_fn so they never become reachable."""
    test_features = _make_test_features()
    monkeypatch.setattr(
        "litellm.proxy._lazy_features.LAZY_FEATURES", test_features, raising=True
    )

    app = _make_app_with_lazy_middleware(features=test_features)

    def trim(target_app: FastAPI) -> None:
        target_app.router.routes = [
            r
            for r in target_app.router.routes
            if not isinstance(r, APIRoute) or r.path.startswith("/data/")
        ]

    restrict_lazy_features_to_component(
        app,
        allowed_path_prefixes=("/data/",),
        allowed_exact_paths=frozenset(),
        re_trim=trim,
    )

    spec = next(s for s in app.user_middleware if s.cls is LazyFeatureMiddleware)
    mw = LazyFeatureMiddleware(app, **spec.kwargs)

    await _hit(mw, "/data/items")

    paths = _route_paths(app)
    assert "/data/items" in paths, "allowed data-plane route should be registered"
    assert not any(
        p.startswith("/admin/") for p in paths
    ), f"admin route leaked despite re-trim: {sorted(p for p in paths if p.startswith('/admin/'))}"


@pytest.mark.asyncio
async def test_admin_only_feature_still_loads_on_management_component(monkeypatch):
    """Symmetric check: a feature whose prefixes are entirely in the management
    allowlist must still load on the management component."""
    test_features = _make_test_features()
    monkeypatch.setattr(
        "litellm.proxy._lazy_features.LAZY_FEATURES", test_features, raising=True
    )

    app = _make_app_with_lazy_middleware(features=test_features)

    def trim(target_app: FastAPI) -> None:
        target_app.router.routes = [
            r
            for r in target_app.router.routes
            if not isinstance(r, APIRoute) or r.path.startswith("/admin/")
        ]

    restrict_lazy_features_to_component(
        app,
        allowed_path_prefixes=("/admin/",),
        allowed_exact_paths=frozenset(),
        re_trim=trim,
    )

    spec = next(s for s in app.user_middleware if s.cls is LazyFeatureMiddleware)
    mw = LazyFeatureMiddleware(app, **spec.kwargs)

    await _hit(mw, "/admin/things")

    assert "/admin/things" in _route_paths(app)


@pytest.mark.asyncio
async def test_helper_raises_when_middleware_not_attached():
    """Calling the helper on an app without LazyFeatureMiddleware is a wiring
    error -- fail loudly, don't silently leave the leak in place."""
    app = FastAPI()  # no middleware added

    with pytest.raises(RuntimeError, match="LazyFeatureMiddleware is not attached"):
        restrict_lazy_features_to_component(
            app,
            allowed_path_prefixes=("/x/",),
            allowed_exact_paths=frozenset(),
            re_trim=lambda _a: None,
        )


def test_re_trim_runs_synchronously_inside_register(monkeypatch):
    """The wrapped register_fn must call ``re_trim`` AFTER the original
    register has finished -- otherwise the new routes wouldn't be present yet
    when the trim runs and nothing would be filtered."""
    test_features = _make_test_features()
    monkeypatch.setattr(
        "litellm.proxy._lazy_features.LAZY_FEATURES", test_features, raising=True
    )

    app = _make_app_with_lazy_middleware(features=test_features)
    saw_route_during_trim: list[bool] = []

    def trim(target_app: FastAPI) -> None:
        saw_route_during_trim.append(
            any(
                isinstance(r, APIRoute) and r.path == "/admin/things"
                for r in target_app.router.routes
            )
        )
        target_app.router.routes = [
            r for r in target_app.router.routes if not isinstance(r, APIRoute)
        ]

    restrict_lazy_features_to_component(
        app,
        allowed_path_prefixes=("/admin/",),
        allowed_exact_paths=frozenset(),
        re_trim=trim,
    )

    spec = next(s for s in app.user_middleware if s.cls is LazyFeatureMiddleware)
    mw = LazyFeatureMiddleware(app, **spec.kwargs)
    asyncio.get_event_loop().run_until_complete(_hit(mw, "/admin/things"))

    assert saw_route_during_trim and saw_route_during_trim[0] is True, (
        "re_trim must observe the newly-registered route, i.e. it must run "
        "after the original register_fn (not before it)"
    )
