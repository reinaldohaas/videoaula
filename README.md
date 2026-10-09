# videoaula — vídeo-aulas com voz clonada, 100 % local

Converte uma apresentação `.pptx` em vídeo-aula Full HD: um clipe por slide, narração pela
voz clonada do professor a partir das **anotações do orador**, legendas sincronizadas palavra
a palavra e avisos no topo (abertura/encerramento). Nada sai da máquina: síntese, alinhamento
e renderização rodam localmente.

Desenvolvido para a disciplina FSC7116 – Mesoescala (UFSC), mas serve para qualquer `.pptx`.

## Como funciona

```
.pptx ──► slides PNG + texto das anotações
              │
              ├─► motor de voz (frase a frase, com a amostra de referência)
              │        reinaldo_haas : Chatterbox Multilingual (zero-shot)
              │        reinaldo_f5   : F5-TTS pt-BR (mais leve, ajustado em português)
              │        antonio/francisca : edge-tts (nuvem, fallback)
              ├─► whisperx: alinhamento forçado → tempo de cada palavra
              ├─► legendas .ass (palavra atual em destaque)
              └─► ffmpeg: clipe por slide → concatenação → video/<aula>_AULA_COMPLETA.mp4
```

Cache inteligente: cada áudio guarda uma assinatura (texto, motor, amostra, parâmetros).
Só regenera o que mudou — editar a anotação de um slide refaz só aquele slide.

## Requisitos

- Python 3.11, GPU NVIDIA (≥ 4 GB de VRAM; 8 GB confortável) — CPU funciona mas leva ~10 min/slide
- `ffmpeg` no PATH
- **Windows**: PowerPoint instalado (exportação dos slides via COM) + `pip install pywin32`
- **Linux/cluster**: LibreOffice (`soffice`) + poppler (`pdftoppm`) + `pip install python-pptx`
- Amostra da sua voz em `voz_referencia/amostra_nova.mp3` (10–30 s, limpa, sem música; o
  arquivo **não** é versionado — veja `.gitignore`)

```bash
mamba create -n voz python=3.11 ffmpeg -c conda-forge
mamba activate voz
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Primeira execução (baixa os modelos para `~/.cache/huggingface`, ~5 GB):

```bash
python gerador_de_video_e_som.py aula.pptx -s 1 --online
```

Depois, sempre offline (padrão):

```bash
python gerador_de_video_e_som.py aula.pptx            # aula inteira
python gerador_de_video_e_som.py aula.pptx -s 3-5     # só alguns slides
python gerador_de_video_e_som.py aula.pptx -v reinaldo_f5   # motor F5-TTS pt-BR
python gerador_de_video_e_som.py aula.pptx --cfg 0.3  # mais fiel ao timbre (Chatterbox)
python gerador_de_video_e_som.py aula.pptx --forcar   # ignora o cache
```

`python gerador_de_video_e_som.py -h` lista todas as opções.

## No cluster (vlab/UFSC)

`videoaula_vlab.ipynb` instala o ambiente, baixa os modelos, testa um slide e submete a aula
inteira como job Slurm. Ajuste os cabeçalhos `#SBATCH` à configuração do cluster.

## Estrutura

```
gerador_de_video_e_som.py   pipeline completo (CLI)
videoaula_vlab.ipynb        instalação e execução no cluster
requirements.txt
voz_referencia/             amostra de voz (ignorada pelo git)
video_work/                 cache: PNGs, áudios, legendas, clipes (ignorado)
video/                      vídeos finais (ignorados)
```

## Aviso

A narração é gerada por IA a partir de uma amostra da voz do autor. Os vídeos trazem
o aviso "voz clonada localmente por programa de computador" na abertura.
