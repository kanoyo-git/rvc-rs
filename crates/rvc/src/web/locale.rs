use axum::extract::Path;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};

macro_rules! dictionaries {
    ($language:expr, $($name:literal),+ $(,)?) => {
        match $language.as_str() {
            $(
                $name => include_str!(concat!("../../resources/locales/", $name, ".json")),
            )+
            _ => return StatusCode::NOT_FOUND.into_response(),
        }
    }
}

pub(super) async fn messages(Path(language): Path<String>) -> Response {
    let json = dictionaries!(
        language, "en_US", "es_ES", "fr_FR", "it_IT", "ja_JP", "ko_KR", "pt_BR", "ru_RU", "tr_TR",
        "zh_CN", "zh_HK", "zh_SG", "zh_TW",
    );
    (
        [(header::CONTENT_TYPE, "application/json; charset=utf-8")],
        json,
    )
        .into_response()
}
