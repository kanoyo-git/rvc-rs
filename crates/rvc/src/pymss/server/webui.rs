use axum::Router;
use axum::extract::Path;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::get;

pub(super) fn add_routes<S: Clone + Send + Sync + 'static>(router: Router<S>) -> Router<S> {
    router
        .route("/ui", get(|| async { Redirect::temporary("/ui/") }))
        .route("/ui/", get(index))
        .route("/ui/assets/{*name}", get(asset))
        .route("/ui/{*page}", get(index))
}

async fn index() -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache"),
            (
                header::CONTENT_SECURITY_POLICY,
                "default-src 'self'; media-src blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            ),
        ],
        include_str!("webui/index.html"),
    )
        .into_response()
}

async fn asset(Path(name): Path<String>) -> Response {
    let (mime, bytes) = match name.as_str() {
        "app.css" => ("text/css; charset=utf-8", include_str!("webui/app.css")),
        "app.js" => (
            "application/javascript; charset=utf-8",
            include_str!("webui/app.js"),
        ),
        _ => return StatusCode::NOT_FOUND.into_response(),
    };
    (
        [
            (header::CONTENT_TYPE, mime),
            (header::CACHE_CONTROL, "no-cache"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        ],
        bytes,
    )
        .into_response()
}
