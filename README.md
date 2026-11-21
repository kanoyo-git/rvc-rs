# NVC Studio (rvc-rs)

Реализация [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) на Rust: конвертация голоса, каверы, разделение вокала и обучение моделей. Работает с моделями RVC v1/v2 и индексами FAISS, без Python.

[Открыть в Colab](https://colab.research.google.com/github/kanoyo-git/rvc-rs/blob/main/NVC-Studio.ru.ipynb) · [English notebook](https://colab.research.google.com/github/kanoyo-git/rvc-rs/blob/main/NVC-Studio.ipynb) · [Релизы](https://github.com/kanoyo-git/rvc-rs/releases)

## Возможности

- **NVC Studio** — веб-интерфейс (`rvc serve`): конвертация файлов и папок, каверы, разделение, обучение, работа с моделями, реалтайм.
- **Каверы** (`rvc cover`): песня делится на минус, основной голос, бэки и реверб, каждый сухой слой конвертируется отдельно, затем микс собирается заново.
- **Разделение вокала**: BS-RoFormer и Mel-Band-RoFormer (вокал, karaoke, снятие реверба), а также каталог PyMSS.
- **Обучение**: предобработка, извлечение признаков (HuBERT Base, ContentVec, SPIN, mHuBERT-147), обучение с продолжением с последнего чекпойнта, построение индекса, TensorBoard, FP16/BF16, несколько GPU.
- **Улучшения качества**: ВЧ-полосы для шипящих (`--hf-bands`), многослойные входы (`layer-aug`), обработка вдохов (`--aperiodic`), голосовой гейт, ограничение уровня.
- **Реалтайм** через аудиоустройства и плагин **VST2/VST3** ([crates/rvc-plugin](crates/rvc-plugin/README.md)).
- **Веса по требованию**: недостающие файлы скачиваются при первом запуске задачи, с прогрессом и проверкой SHA-256.

## Google Colab

Блокноты [NVC-Studio.ru.ipynb](NVC-Studio.ru.ipynb) и [NVC-Studio.ipynb](NVC-Studio.ipynb) скачивают готовую сборку (~10 МБ) из релиза и запускают Studio на PyTorch, который уже есть в Colab. Ссылка открывается через Gradio, cloudflared или localtunnel, вход по логину `rvc` и паролю. Если туннель недоступен, раздел «Без интерфейса» в конце блокнота выполняет те же операции в ячейках.

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

Обучение голоса из CLI:

```bash
rvc preprocess --input dataset/ --experiment logs/my-voice --sample-rate 40000
rvc extract --experiment logs/my-voice --embedder mhubert147 --f0-method rmvpe
rvc train --experiment logs/my-voice --embedder mhubert147 --epochs 200 --save-weights \
  --pretrained-g assets/pretrained_v2/NVC-Pretrained/f0G40k.pth \
  --pretrained-d assets/pretrained_v2/NVC-Pretrained/f0D40k.pth
rvc train-index --experiment logs/my-voice
```

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
**NVC-Pretrained** (GPL-3.0) — претрейн 40 кГц на признаках mHuBERT-147, дообученный на русской речи и пении, а также на крике, скриме и гроуле. Голоса от него обучайте с `--embedder mhubert147`, без `--layer-aug`.

## Документация

- [bench/RESULTS.md](bench/RESULTS.md) — сравнение с Python RVC по скорости и памяти
- [bench/README.md](bench/README.md) — как повторить замеры
- [crates/rvc-plugin/README.md](crates/rvc-plugin/README.md) — плагин VST

## Лицензия

[MIT](LICENSE). Исходные проекты и лицензии используемых компонентов перечислены в [THIRD_PARTY.md](THIRD_PARTY.md).
