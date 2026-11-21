# Сторонние компоненты

## Retrieval-based-Voice-Conversion-WebUI

RVC-RS — самостоятельная реализация функций исходного проекта на Rust. Исходная версия служит эталоном для проверки поведения и сравнительных замеров. В Rust-версию перенесены конфигурации моделей и словари интерфейса.

Авторы, указанные в исходной лицензии, — liujing04, 源文雨 и Ftps. [Исходный проект](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) · [Эталонный коммит `81eed5e`](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/commit/81eed5e8f68b6bed1789f682fe78cdd324495afc) · [Лицензия MIT](LICENSE) · [Уведомления о связанных проектах](licenses/related-projects.txt)

## NVC Studio

Из NVC перенесены оформление Studio, переводы RU/EN, значок и каталог предобученных весов. Интерфейс подключён к нативному API RVC-RS. Реализации NMVPE, подбора регистра, многомасштабной mel-потери и NVC-TS опираются на исходный код NVC. Rust RVC читает пулы и метаданные голосовых моделей NVC и использует тот же набор альтернативных эмбеддеров. [NVC](https://github.com/kanoyo-git/NVC) · [Лицензия MIT](LICENSE)

## PyMSS и PyMSS Core

Оригинальные компоненты KitsuneX07 использовались для проверки разделения дорожек. В Rust-версию перенесены каталог моделей и профили UVR. [Лицензия PyMSS](licenses/pymss.txt) · [Лицензия PyMSS Core](licenses/pymss-core.txt)

## RVC Realtime VST

При переносе плагина сохранена совместимость форматов пресетов и настроек исходного RVC Realtime VST от RVC-BOSS. [Лицензия RVC Realtime VST](licenses/rvc-realtime-vst.txt)
