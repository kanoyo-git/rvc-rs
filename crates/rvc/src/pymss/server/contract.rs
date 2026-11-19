use axum::{
    Router,
    http::header,
    response::{Html, IntoResponse},
    routing::get,
};

const OPENAPI: &str = include_str!("contract/openapi.json");
const REFERENCE: &str = include_str!("contract/index.html");

pub(super) fn add_routes<S: Clone + Send + Sync + 'static>(router: Router<S>) -> Router<S> {
    router
        .route("/openapi.json", get(schema))
        .route("/docs", get(reference))
        .route("/redoc", get(reference))
}

async fn schema() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/json")], OPENAPI)
}

async fn reference() -> Html<&'static str> {
    Html(REFERENCE)
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::OPENAPI;

    #[test]
    fn contract_covers_service_routes_and_pcm_transports() {
        let schema: Value = serde_json::from_str(OPENAPI).unwrap();
        let paths = schema["paths"].as_object().unwrap();
        let operations = [
            ("/health", "get"),
            ("/v1/models", "get"),
            ("/v1/models/{model}", "get"),
            ("/v1/catalog/models", "get"),
            ("/v1/catalog/models/{model}", "get"),
            ("/v1/models/load", "post"),
            ("/v1/models/download", "post"),
            ("/v1/download-source", "get"),
            ("/v1/download-source", "post"),
            ("/v1/server/info", "get"),
            ("/v1/audio/separations", "post"),
        ];
        for (path, method) in operations {
            assert!(paths[path][method]["responses"]["200"].is_object());
        }
        let separation = &paths["/v1/audio/separations"]["post"];
        let request = &separation["requestBody"]["content"];
        assert!(request["application/json"].is_object());
        assert!(request["application/octet-stream"].is_object());
        assert!(separation["responses"]["200"]["content"]["application/zip"].is_object());
    }
}
