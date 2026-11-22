# NVC Studio (rvc-rs)

Реализация [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) на Rust: конвертация голоса, каверы, разделение вокала и обучение моделей. Работает с моделями RVC v1/v2 и индексами FAISS, без Python.

[Открыть в Colab](https://colab.research.google.com/github/kanoyo-git/rvc-rs/blob/main/NVC-Studio.ru.ipynb) · [English notebook](https://colab.research.google.com/github/kanoyo-git/rvc-rs/blob/main/NVC-Studio.ipynb) · [Релизы](https://github.com/kanoyo-git/rvc-rs/releases)

## Возможности

- **NVC Studio** — веб-интерфейс (`rvc serve`): конвертация файлов и папок, каверы, разделение, обучение, работа с моделями, реалтайм.
- **Каверы** (`rvc cover`): песня делится на минус, основной голос, бэки и реверб, каждый сухой слой конвертируется отдельно, затем микс собирается заново.
- Во вкладке **Кавер** Studio запоминает результат каждой песни. Если заменить один файл в подборке, при прежнем голосе и настройках обработается только этот файл. Для нового дубля отключите «Использовать кеш каверов». Кеш хранится в `TEMP/cover-cache` внутри рабочей папки Studio.
- Во время отправки аудио нажмите **Отменить загрузку**. Это работает для одного файла, пакета, кавера, разделения, профиля регистра и датасета. При отмене замены датасета останется выбран предыдущий.
- **Разделение вокала**: BS-RoFormer и Mel-Band-RoFormer (вокал, karaoke, снятие реверба), а также каталог PyMSS.
- **Обучение**: предобработка, извлечение признаков (HuBERT Base, ContentVec, SPIN, mHuBERT-147), обучение с продолжением с последнего чекпойнта, построение индекса, TensorBoard, FP16/BF16, несколько GPU.
- В **Библиотеке** можно вставить общедоступную ссылку Google Диска, MEGA, Яндекс Диска, MediaFire, Hugging Face или прямой URL файла. Голос можно импортировать из .pth или ZIP, претрейн из одного ZIP с парой весов или по отдельным ссылкам на генератор и дискриминатор.
- В разделе **Обучение** можно перетащить ZIP, один аудиофайл или несколько записей. Для разных дикторов загрузите ZIP с отдельной папкой на каждого.
- **Улучшения качества**: ВЧ-полосы для шипящих (`--hf-bands`), многослойные входы (`layer-aug`), обработка вдохов (`--aperiodic`), голосовой гейт, ограничение уровня.
- При конвертации старых RVC-моделей Studio по умолчанию использует шипящие самого генератора. Модели NVC с нулевой F0 на глухих кадрах или с входом ВЧ-полос получают огибающую исходных согласных и вдохов. Поле «Вдохи и шипящие» и параметр `--aperiodic` позволяют задать силу вручную.
- **Реалтайм** через аудиоустройства и плагин **VST2/VST3** ([crates/rvc-plugin](crates/rvc-plugin/README.md)).
- **Веса по требованию**: недостающие файлы скачиваются при первом запуске задачи, с прогрессом и проверкой SHA-256.

После каждой завершённой эпохи Studio выводит её длительность и примерное время до конца обучения. Оценка использует среднее время эпох текущего запуска, в том числе при продолжении с чекпойнта. При повторном открытии адреса работающей Studio восстанавливаются настройки и журналы задач.

## Google Colab

Блокноты [NVC-Studio.ru.ipynb](NVC-Studio.ru.ipynb) и [NVC-Studio.ipynb](NVC-Studio.ipynb) скачивают готовую сборку (~12 МБ) из релиза и запускают Studio на PyTorch, который уже есть в Colab. Ссылка открывается через Gradio, cloudflared или localtunnel, вход по логину `rvc` и паролю. Если туннель недоступен, раздел «Без интерфейса» в конце блокнота выполняет те же операции в ячейках.

## Сборка

Нужны Rust, компилятор C++, CMake, BLAS, FFmpeg и `curl`, а также [LibTorch 2.7.0](https://pytorch.org/get-started/locally/) (CPU или CUDA). На Linux ещё понадобятся `pkg-config` и заголовки ALSA, X11 и OpenGL.

```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo build --release --locked -p rvc
```

На Windows добавьте `%LIBTORCH%\lib` и FFmpeg в `PATH`; `go-webui.bat` запускает Studio. Для построения индексов на Windows и macOS нужна библиотека FAISS 1.12.0 с C API (путь в `FAISS_LIB_DIR`), на Linux FAISS собирается сам.

## Запуск

```bash
rvc serve --open                                   # NVC Studio в браузере
rvc infer --model assets/weights/voice.pth --input song.wav --output out/
rvc cover --model assets/weights/voice.pth --input song.flac --output covers/
rvc separate --input song.flac --output stems/ --preset vocals-bs-roformer-368
```

Для обучения голоса с опубликованным NVC-Pretrained используйте ContentVec при извлечении признаков и обучении.

```bash
rvc preprocess --input dataset/ --experiment logs/my-voice --sample-rate 40000
rvc extract --experiment logs/my-voice --embedder hubert_base --f0-method rmvpe
rvc train --experiment logs/my-voice --embedder hubert_base --sample-rate 40000 \
  --epochs 60 --batch-size 8 --save-every 60 --save-weights --unvoiced-f0 zero \
  --pretrained-g assets/pretrained_v2/NVC-Pretrained-Rage-ContentVec/f0G40k.pth \
  --pretrained-d assets/pretrained_v2/NVC-Pretrained-Rage-ContentVec/f0D40k.pth
rvc train-index --experiment logs/my-voice
```

Выберите `--f0-method rmcpe` в `rvc extract`, `rvc infer` или `rvc realtime`, чтобы объединить прогнозы RMVPE и FCPE с проверкой периода по звуку. Для экстракции и конверсии укажите один и тот же метод. CLI и Studio загружают обе модели при первом использовании. [Устройство и измерения RMCPE](bench/RMCPE.md)

`rvc <команда> --help` показывает все параметры. Для доступа с другого устройства: `rvc serve --listen 0.0.0.0`, через внешнюю ссылку: `rvc serve --colab` (Gradio) или `rvc serve --public-url <https-адрес туннеля>`.

## Модели и веса

Голосовые модели кладите в `assets/weights/`, индексы в `assets/indices/`. Остальное Studio и CLI скачивают сами при первой задаче, которой это нужно:

| Что | Источник |
| --- | --- |
| RMVPE, предобученные RVC v1/v2 | [Politrees/RVC_resources](https://huggingface.co/Politrees/RVC_resources) |
| HuBERT Base, `mute.zip`, модели разделения | [релиз resources](https://github.com/kanoyo-git/rvc-rs/releases/tag/resources) |
| NVC-Pretrained | [kanoyo/NVC-Pretrained](https://huggingface.co/kanoyo/NVC-Pretrained) |
| NMVPE | [kanoyo/nvc_resources](https://huggingface.co/kanoyo/nvc_resources) |
| mHuBERT-147, ContentVec, SPIN | Hugging Face (utter-project, IAHispano/Applio) |
| FCPE | пакет [torchfcpe 0.0.4](https://pypi.org/project/torchfcpe/0.0.4/) на PyPI |

**NVC-Pretrained** (GPL-3.0) использует ContentVec (`hubert_base`) и обучен на вокале, включая рейдж. В Studio и Colab выберите его для обучения на 40 кГц. Основной эмбеддер в Studio подписан **ContentVec (RVC)**. Кодировщик в разделе «Качество и эксперименты» участвует в обучении при положительном весе фонетической потери. По умолчанию он использует тот же эмбеддер. Генератор включает парные входы через метаданные весов. Новая пара скачивается в `assets/pretrained_v2/NVC-Pretrained-Rage-ContentVec/`. Каталог `assets/pretrained_v2/NVC-Pretrained/` хранит прежнюю версию для экспериментов с mHuBERT-147.

## Документация

- [bench/RESULTS.md](bench/RESULTS.md) — сравнение с Python RVC по скорости и памяти
- [bench/README.md](bench/README.md) — как повторить замеры
- [crates/rvc-plugin/README.md](crates/rvc-plugin/README.md) — плагин VST

## Лицензия

[MIT](LICENSE). Исходные проекты и лицензии используемых компонентов перечислены в [THIRD_PARTY.md](THIRD_PARTY.md).
