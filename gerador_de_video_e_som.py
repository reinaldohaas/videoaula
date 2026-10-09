"""
Gerador de Vídeo e Som para Aulas e Apresentações
Converte apresentações PowerPoint (.pptx) diretamente em vídeo-aulas Full HD,
com narração pela voz clonada LOCALMENTE do Prof. Reinaldo Haas
(Chatterbox Multilingual + alinhamento whisperx) ou vozes edge-tts como fallback,
legendas sincronizadas palavra por palavra em tempo real (WordBoundary),
avisos visuais no topo de até 5s, legendas 30% maiores,
áudio e legendas baseados EXCLUSIVAMENTE nas anotações do PPT,
e salvamento automático na pasta 'video'.
"""

import argparse
import asyncio
import json
import os
import pathlib
import re
import subprocess
import sys
import time


def _cpus_disponiveis() -> int:
    """Núcleos realmente utilizáveis: cota do cgroup (contêiner/cluster) ou affinity, não o total do nó."""
    n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    for caminho in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            partes = open(caminho).read().split()
            if partes[0] != "max" and int(partes[0]) > 0:
                periodo = int(partes[1]) if len(partes) > 1 else int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
                n = min(n, max(1, int(int(partes[0]) / periodo)))
                break
        except Exception:
            continue
    return max(1, n)


_N_CPU = _cpus_disponiveis()
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(min(_N_CPU, 16)))

import torch
torch.set_num_threads(min(_N_CPU, 16))

TOOL_DIR = pathlib.Path(__file__).resolve().parent
VOZES_DIR = TOOL_DIR / "vozes"

# Textos padrão dos avisos no topo da tela (exibidos por até 5 segundos)
DEFAULT_AVISO_INICIO = (
    r"{\c&H00D7FF&\b1}UFSC · PROF. REINALDO HAAS\N"
    r"{\c&HFFFFFF&\b1}VOZ CLONADA LOCALMENTE POR PROGRAMA DE COMPUTADOR"
)

DEFAULT_AVISO_FIM = (
    r"{\c&H00D7FF&\b1}UFSC · PROF. REINALDO HAAS\N"
    r"{\c&HFFFFFF&\b1}FIM DA AULA · BONS ESTUDOS COM OS EXERCÍCIOS!"
)

# -------------------------------------------------------------
# Normalização Fonética de Texto para Fala
# -------------------------------------------------------------
def normalizar_para_fala(texto: str) -> str:
    t = texto
    # Remove número isolado de página no final de notas
    t = re.sub(r'\s+\d+\s*$', '', t)
    t = re.sub(r'\bUFSC\b', 'U F S C', t)
    t = re.sub(r'\bSALLJ\b', 'S A L L Jota', t)

    # Notações matemáticas comuns
    t = t.replace('k⁻³', 'k a menos três')
    t = t.replace('k⁻⁵ᐟ³', 'k a menos cinco terços')
    t = t.replace('k⁻⁵/³', 'k a menos cinco terços')
    t = re.sub(r'\bkm\b', 'quilômetros', t)
    t = re.sub(r'\bm/s\b', 'metros por segundo', t)

    t = re.sub(r'\bCapítulo\s+1\b', 'Capítulo um', t, flags=re.IGNORECASE)
    t = re.sub(r'\bCapítulo\s+2\b', 'Capítulo dois', t, flags=re.IGNORECASE)
    t = re.sub(r'\bCapítulo\s+3\b', 'Capítulo três', t, flags=re.IGNORECASE)
    t = re.sub(r'\bCapítulo\s+4\b', 'Capítulo quatro', t, flags=re.IGNORECASE)
    t = re.sub(r'\bCapítulo\s+5\b', 'Capítulo cinco', t, flags=re.IGNORECASE)

    def sub_secao(m):
        n1 = m.group(1)
        n2 = m.group(2)
        n_map = {
            '0': 'zero', '1': 'um', '2': 'dois', '3': 'três', '4': 'quatro',
            '5': 'cinco', '6': 'seis', '7': 'sete', '8': 'oito', '9': 'nove', '10': 'dez',
            '11': 'onze', '12': 'doze', '13': 'treze', '14': 'quatorze', '15': 'quinze'
        }
        w1 = n_map.get(n1, n1)
        w2 = n_map.get(n2, n2)
        return f"{w1} ponto {w2}"

    t = re.sub(r'\b(\d+)\.(\d+)\b', sub_secao, t)

    t = re.sub(r'\bTab\.\s*', 'Tabela ', t, flags=re.IGNORECASE)
    t = re.sub(r'\bFig\.\s*', 'Figura ', t, flags=re.IGNORECASE)
    t = re.sub(r'\bEq\.\s*', 'Equação ', t, flags=re.IGNORECASE)

    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# -------------------------------------------------------------
# Utilitários de Legendas ASS Full HD (com Top Banner e Fonte +30%)
# -------------------------------------------------------------
def format_ass_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h:01d}:{m:02d}:{s:02d}.{cs:02d}"

def reconciliar_palavras_com_pontuacao(words_timing: list, texto_original: str) -> list:
    """
    Restaura as pontuações originais (.,:;!?-), acentos e caixa alta/baixa do texto original,
    pois o Edge-TTS WordBoundary retorna palavras isoladas limpas de pontuação.
    """
    if not texto_original or not words_timing:
        return words_timing

    tokens_originais = texto_original.split()

    def limpar(s):
        return re.sub(r'[^\w]', '', s).lower()

    resultado = []
    t_idx = 0
    num_tokens = len(tokens_originais)

    for item in words_timing:
        w_text, s, e = item
        w_clean = limpar(w_text)

        encontrou = False
        for k in range(t_idx, min(t_idx + 8, num_tokens)):
            tok_clean = limpar(tokens_originais[k])
            if tok_clean == w_clean:
                resultado.append((tokens_originais[k], s, e))
                t_idx = k + 1
                encontrou = True
                break
        if not encontrou:
            resultado.append((w_text, s, e))

    return resultado

def montar_chunks_naturais(words_timing: list, max_words: int = 7, max_chars: int = 42) -> list:
    """
    Agrupa palavras em frases naturais respeitando pontuação e limites de largura,
    evitando quebras no meio de ideias e garantindo frases legíveis na tela.
    """
    chunks = []
    cur = []
    cur_len = 0
    for item in words_timing:
        w_text, s, e = item
        cur.append(item)
        cur_len += len(w_text) + 1
        tem_pontuacao = any(p in w_text for p in ['.', '!', '?', ';', ':', '—'])
        tem_virgula = any(p in w_text for p in [','])
        if (tem_pontuacao and len(cur) >= 3) or (tem_virgula and len(cur) >= 4) or len(cur) >= max_words or cur_len >= max_chars:
            chunks.append(cur)
            cur = []
            cur_len = 0
    if cur:
        if chunks and len(cur) <= 2:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    return chunks

def gerar_ass_wordboundary(
    words_timing: list,
    texto_original: str = None,
    chunk_size: int = 7,
    aviso_topo: str = None,
    duracao_aviso: float = 5.0,
    total_dur: float = 10.0,
    offset_sincronia: float = -0.12
) -> str:
    """
    Gera legendas ASS contínuas com pontuação completa, sem piscar e sem sobrepor os slides.
    A frase inteira permanece visível na tela em branco com sua pontuação original,
    e apenas a palavra atual muda para amarelo ouro (&H00D7FF&) no momento exato em que é pronunciada.
    MarginV ajustada para 24 para nunca sobrepor o rodapé/gráficos do slide.
    """
    # 1. Restaura pontuações e acentos do texto original
    if texto_original:
        words_timing = reconciliar_palavras_com_pontuacao(words_timing, texto_original)

    # 2. Aplica offset de sincronização para eliminar atraso perceptual
    ajustados = []
    for w, s, e in words_timing:
        ns = max(0.0, s + offset_sincronia)
        ne = max(ns + 0.05, e + offset_sincronia)
        ajustados.append((w, ns, ne))

    chunks = montar_chunks_naturais(ajustados, max_words=chunk_size, max_chars=42)

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: AulaStyle,{FONTE_LEGENDA},50,&H00FFFFFF,&H0000D7FF,&H00000000,&HB0000000,-1,0,0,0,100,100,0,0,1,3,2,2,60,60,18,1",
        f"Style: AvisoTopoStyle,{FONTE_LEGENDA},52,&H00FFFFFF,&H0000D7FF,&H00000000,&HB0000000,-1,0,0,0,100,100,0,0,1,3,2,8,60,60,45,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]

    if aviso_topo:
        tempo_fim_aviso = min(duracao_aviso, max(total_dur, 1.0))
        t_start_str = format_ass_time(0.0)
        t_end_str = format_ass_time(tempo_fim_aviso)
        aviso_ass = aviso_topo.replace("\n", r"\N")
        lines.append(f"Dialogue: 1,{t_start_str},{t_end_str},AvisoTopoStyle,,0,0,0,,{aviso_ass}")

    for c_idx, chunk in enumerate(chunks):
        if not chunk:
            continue
        next_chunk_start = chunks[c_idx + 1][0][1] if c_idx + 1 < len(chunks) else total_dur
        chunk_start = chunk[0][1]
        last_word_end = chunk[-1][2]

        # Estende a frase na tela até o próximo chunk começar se a pausa for de até 1.2s,
        # garantindo que a frase NUNCA pisque ou desapareça entre falas normais
        if next_chunk_start - last_word_end <= 1.2:
            chunk_end = next_chunk_start
        else:
            chunk_end = min(next_chunk_start, last_word_end + 0.8)

        cur_t = chunk_start
        for w_idx, (w_text, w_start, w_end) in enumerate(chunk):
            if w_idx + 1 < len(chunk):
                next_t = chunk[w_idx + 1][1]
            else:
                next_t = min(w_end, chunk_end)

            if next_t < cur_t:
                next_t = cur_t + 0.05

            parts = []
            for j, (other_w, _, _) in enumerate(chunk):
                u_text = other_w.upper()
                if j == w_idx:
                    parts.append(r"{\c&H00D7FF&\b1}" + u_text + r"{\c&HFFFFFF&\b0}")
                else:
                    parts.append(u_text)
            txt_line = " ".join(parts)
            lines.append(f"Dialogue: 0,{format_ass_time(cur_t)},{format_ass_time(next_t)},AulaStyle,,0,0,0,,{txt_line}")
            cur_t = next_t

        # Período de repouso da frase completa em branco até o próximo chunk
        if chunk_end > cur_t + 0.05:
            full_line_white = " ".join(w[0].upper() for w in chunk)
            lines.append(f"Dialogue: 0,{format_ass_time(cur_t)},{format_ass_time(chunk_end)},AulaStyle,,0,0,0,,{full_line_white}")

    return "\n".join(lines)

# -------------------------------------------------------------
# Extração Direta e Otimizada do PPTX (Somente Anotações)
# -------------------------------------------------------------
def _montar_item(i: int, img_path: pathlib.Path, notas: str, textos_shape: list) -> dict:
    conteudo_base = notas.strip() if notas and notas.strip() else " ".join(textos_shape)
    conteudo_normalizado = normalizar_para_fala(conteudo_base)
    return {
        "slide": i,
        "imagem": str(img_path.name),
        "texto_fala": conteudo_normalizado,
        "texto_original": conteudo_base,
        "texto_exibicao": conteudo_normalizado,
    }


def extrair_e_montar_roteiro_linux(
    pptx_path: pathlib.Path,
    slides_img_dir: pathlib.Path,
    slides_filtro: list = None
) -> list:
    """Linux/macOS (cluster): LibreOffice headless -> PDF -> pdftoppm (PNG 1920x1080);
    anotações do orador via python-pptx. Requer: libreoffice (soffice), poppler-utils, pip install python-pptx."""
    from pptx import Presentation
    import shutil

    slides_img_dir.mkdir(parents=True, exist_ok=True)
    pres = Presentation(str(pptx_path))
    total_slides = len(pres.slides)
    alvos = [s for s in range(1, total_slides + 1) if not slides_filtro or s in slides_filtro]
    print(f"Apresentacao: {pptx_path.name} (Total: {total_slides} slides, preparando: {len(alvos)} slide(s)) [LibreOffice]...")

    pptx_mtime = pptx_path.stat().st_mtime
    faltam = [i for i in alvos
              if not (slides_img_dir / f"slide_{i:03d}.png").exists()
              or (slides_img_dir / f"slide_{i:03d}.png").stat().st_size == 0
              or (slides_img_dir / f"slide_{i:03d}.png").stat().st_mtime < pptx_mtime]
    if faltam:
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if not soffice:
            raise RuntimeError("LibreOffice (soffice) não encontrado no PATH. No cluster: module load libreoffice, "
                               "ou instale com conda: mamba install -c conda-forge libreoffice poppler")
        pdf_path = slides_img_dir / (pptx_path.stem + ".pdf")
        if not pdf_path.exists() or pdf_path.stat().st_mtime < pptx_mtime:
            print("  Convertendo PPTX -> PDF com LibreOffice (uma vez)...")
            subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(slides_img_dir), str(pptx_path)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
            if not pdf_path.exists():
                raise RuntimeError("LibreOffice não gerou o PDF.")
        if not shutil.which("pdftoppm"):
            raise RuntimeError("pdftoppm (poppler-utils) não encontrado. mamba install -c conda-forge poppler")
        for i in faltam:
            tmp_prefix = slides_img_dir / f"_tmp_{i:03d}"
            subprocess.run(["pdftoppm", "-png", "-r", "144", "-f", str(i), "-l", str(i),
                            "-scale-to-x", "1920", "-scale-to-y", "1080", str(pdf_path), str(tmp_prefix)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            gerados = sorted(slides_img_dir.glob(f"_tmp_{i:03d}*.png"))
            if not gerados:
                raise RuntimeError(f"pdftoppm não gerou a página {i}.")
            gerados[0].replace(slides_img_dir / f"slide_{i:03d}.png")
            for g in gerados[1:]:
                g.unlink()
            print(f"  Slide {i:03d} exportado para PNG.")
    for i in alvos:
        if i not in faltam:
            print(f"  Slide {i:03d} (imagem ja existe em cache).")

    roteiro = []
    for i in alvos:
        slide = pres.slides[i - 1]
        img_path = slides_img_dir / f"slide_{i:03d}.png"
        notas = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notas = slide.notes_slide.notes_text_frame.text.strip()
        textos_shape = [sh.text_frame.text.strip() for sh in slide.shapes
                        if sh.has_text_frame and sh.text_frame.text.strip()]
        roteiro.append(_montar_item(i, img_path, notas, textos_shape))
    return roteiro


def extrair_e_montar_roteiro(
    pptx_path: pathlib.Path,
    slides_img_dir: pathlib.Path,
    slides_filtro: list = None
) -> list:
    if sys.platform != "win32":
        return extrair_e_montar_roteiro_linux(pptx_path, slides_img_dir, slides_filtro)
    slides_img_dir.mkdir(parents=True, exist_ok=True)
    import win32com.client

    pptx_abs = str(pptx_path.resolve())
    ppt_app = win32com.client.Dispatch("PowerPoint.Application")
    roteiro = []
    try:
        pres = ppt_app.Presentations.Open(pptx_abs, ReadOnly=True, Untitled=False, WithWindow=False)
        total_slides = pres.Slides.Count

        alvos = [s for s in range(1, total_slides + 1) if not slides_filtro or s in slides_filtro]
        print(f"Apresentacao: {pptx_path.name} (Total: {total_slides} slides, preparando: {len(alvos)} slide(s))...")

        for i in alvos:
            slide = pres.Slides(i)
            img_path = slides_img_dir / f"slide_{i:03d}.png"

            # Cache inteligente: reexporta imagem se não existir ou se o PPTX foi modificado depois
            pptx_mtime = pptx_path.stat().st_mtime
            if not img_path.exists() or img_path.stat().st_size == 0 or img_path.stat().st_mtime < pptx_mtime:
                slide.Export(str(img_path), "PNG", 1920, 1080)
                print(f"  Slide {i:03d} exportado para PNG.")
            else:
                print(f"  Slide {i:03d} (imagem ja existe em cache).")

            # Extrai EXCLUSIVAMENTE o texto das anotações do orador
            notas = ""
            if slide.HasNotesPage:
                for s in slide.NotesPage.Shapes:
                    if s.HasTextFrame and s.TextFrame.HasText:
                        txt = s.TextFrame.TextRange.Text.strip()
                        # Ignora se for apenas o número da página
                        if txt and not txt.isdigit():
                            notas += " " + txt
            notas = notas.strip()

            # Se porventura o slide não tiver anotações, usa o texto do slide como fallback
            if notas:
                conteudo_base = notas
            else:
                textos_shape = []
                for s in slide.Shapes:
                    if s.HasTextFrame and s.TextFrame.HasText:
                        t = s.TextFrame.TextRange.Text.strip()
                        if t:
                            textos_shape.append(t)
                conteudo_base = " ".join(textos_shape)

            conteudo_normalizado = normalizar_para_fala(conteudo_base)

            roteiro.append({
                "slide": i,
                "imagem": str(img_path.name),
                "texto_fala": conteudo_normalizado,
                "texto_original": conteudo_base,
                "texto_exibicao": conteudo_normalizado
            })

        pres.Close()
    finally:
        try:
            ppt_app.Quit()
        except Exception:
            pass
    return roteiro

# -------------------------------------------------------------
# Motor de Síntese com Clonagem de Voz (100% local)
# -------------------------------------------------------------
# Voz "reinaldo_haas": Chatterbox Multilingual (clonagem zero-shot a partir da
#   amostra de referência) + whisperx para recuperar os tempos por palavra.
# Vozes "antonio"/"francisca": edge-tts (nuvem Microsoft), mantido como fallback.
#
# Dependências extras:  pip install chatterbox-tts whisperx
#
AMOSTRA_REF_PADRAO = TOOL_DIR / "voz_referencia" / "amostra_nova.mp3"
# Fonte das legendas: Segoe UI no Windows; no Linux (cluster) usa DejaVu Sans, que vem com o libass/fontconfig
FONTE_LEGENDA = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"

# Ajustes de clonagem (Chatterbox)
CB_EXAGGERATION = 0.45      # 0.3 = mais neutro/didático, 0.7 = mais expressivo
CB_CFG_WEIGHT = 0.4         # menor = mais fiel ao timbre da amostra
CB_TEMPERATURE = 0.7
PAUSA_ENTRE_FRASES = 0.35   # segundos de silêncio inserido entre frases
WHISPERX_MODEL = "medium"   # "small" se a GPU for modesta; "large-v3" se sobrar VRAM

# F5-TTS em português brasileiro (voz "reinaldo_f5") — modelo menor, rápido em GPUs de 4 GB
F5_REPO = "traderpedroso/F5-TTS-BRAZILIAN-PORTUGUESE"
F5_ARQUITETURAS = ("F5TTS_Base", "F5TTS_v1_Base")   # tenta nesta ordem
F5_NFE_STEP = 32          # passos do flow matching: 16 = mais rápido, 48 = mais qualidade
F5_CFG = 2.0              # força de guidance (1.5–2.5)
F5_SPEED = 0.95           # ritmo da fala (1.0 = neutro; < 1 = mais pausado, bom para aula)
F5_REF_MAX_S = 12         # F5 rende melhor com referência curta (8–12 s)

VOZES_CLONADAS = ("reinaldo_haas", "reinaldo_f5")

# Renderização dos clipes (imagem parada + legendas)
FPS_CLIPE = 10            # 10 fps: legendas palavra a palavra continuam precisas; 25 só encarece o x264
FFMPEG_THREADS = 4        # por clipe; com --paralelo N o total é N*4


def dividir_em_frases(texto: str) -> list:
    """Quebra o texto em frases; clonagem zero-shot rende melhor em trechos curtos."""
    partes = re.split(r'(?<=[.!?;:])\s+', texto.strip())
    frases = []
    for p in partes:
        p = p.strip()
        if not p:
            continue
        # Frases muito longas (> ~220 chars) são quebradas em vírgulas
        if len(p) > 220:
            sub = re.split(r'(?<=,)\s+', p)
            buf = ""
            for s in sub:
                if len(buf) + len(s) > 220 and buf:
                    frases.append(buf.strip())
                    buf = s
                else:
                    buf = (buf + " " + s).strip()
            if buf:
                frases.append(buf.strip())
        else:
            frases.append(p)
    return frases


def _preencher_tempos_faltantes(palavras: list, total_dur: float) -> list:
    """whisperx não marca tempo em alguns tokens (números, siglas); interpola."""
    n = len(palavras)
    out = [[w, s, e] for (w, s, e) in palavras]
    for i in range(n):
        if out[i][1] is not None and out[i][2] is not None:
            continue
        # Procura vizinho anterior e posterior com tempo
        prev_e = 0.0
        for j in range(i - 1, -1, -1):
            if out[j][2] is not None:
                prev_e = out[j][2]
                break
        next_s, k_next = total_dur, n
        for j in range(i + 1, n):
            if out[j][1] is not None:
                next_s, k_next = out[j][1], j
                break
        # Quantas palavras sem tempo há neste buraco (inclusive esta)
        j0 = i
        while j0 > 0 and out[j0 - 1][1] is None:
            j0 -= 1
        n_gap = k_next - j0
        passo = (next_s - prev_e) / max(n_gap, 1)
        pos = i - j0
        out[i][1] = prev_e + passo * pos
        out[i][2] = prev_e + passo * (pos + 1)
    return [(w, float(s), float(e)) for (w, s, e) in out]


class MotorVoz:
    def __init__(self, nome_voz: str = "reinaldo_haas", amostra_ref: pathlib.Path = None,
                 cfg_weight: float = CB_CFG_WEIGHT, exaggeration: float = CB_EXAGGERATION,
                 permitir_cpu: bool = False, usar_fp16: bool = False):
        self.nome_voz = nome_voz
        self.cfg_weight = cfg_weight
        self.exaggeration = exaggeration
        self.usar_fp16 = usar_fp16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = None
        self.align_model = None
        self.align_meta = None
        self.ref_wav = None

        self.sr = 24000
        if nome_voz in VOZES_CLONADAS:
            if self.device == "cpu" and not permitir_cpu:
                raise RuntimeError(
                    "GPU nao detectada (torch.cuda.is_available() == False). Na CPU cada slide leva ~10 min.\n"
                    "  Verifique: python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"\n"
                    "  Se sair '+cpu', reinstale o torch com CUDA (ex.: --index-url https://download.pytorch.org/whl/cu126).\n"
                    "  Para rodar mesmo assim na CPU, use --permitir-cpu."
                )
            amostra = pathlib.Path(amostra_ref) if amostra_ref else AMOSTRA_REF_PADRAO
            if not amostra.exists():
                raise FileNotFoundError(f"Amostra de referência não encontrada: {amostra}")
            self.ref_wav = self._preparar_referencia(amostra)

            if nome_voz == "reinaldo_haas":
                print(f"Carregando Chatterbox Multilingual ({self.device})...")
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self.tts = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
                self.sr = self.tts.sr
                # Codifica a amostra de referência uma única vez (por padrão o generate() refaz isso a cada frase)
                try:
                    self.tts.prepare_conditionals(str(self.ref_wav), exaggeration=exaggeration)
                    self.conds_prontos = True
                except Exception as e:
                    print(f"  (prepare_conditionals indisponível: {str(e)[:60]} — referência será recodificada por frase)")
                    self.conds_prontos = False
            else:
                self.ref_wav, self.ref_text = self._preparar_referencia_f5(self.ref_wav)
                self._carregar_f5()

            # Alinhamento roda na CPU de propósito: é barato e libera ~1,2 GB de VRAM para o Chatterbox.
            # GPUs com pouca memória (<= 6 GB) caem em "memória compartilhada" do Windows e ficam 10x mais lentas.
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if self.device == "cuda" else 0
            self.align_device = "cuda" if vram_gb >= 12 else "cpu"
            print(f"Carregando modelo de alinhamento whisperx (pt) na {self.align_device.upper()}...")
            import whisperx
            self.align_model, self.align_meta = whisperx.load_align_model(
                language_code="pt", device=self.align_device
            )
            if nome_voz == "reinaldo_haas":
                print(f"Voz clonada ativa (Chatterbox) a partir de: {amostra.name} (cfg={cfg_weight}, exag={exaggeration})")
            else:
                print(f"Voz clonada ativa (F5-TTS pt-BR) a partir de: {amostra.name}")
                print(f"  Transcrição da referência: \"{self.ref_text[:100]}{'...' if len(self.ref_text) > 100 else ''}\"")

    def assinatura(self) -> dict:
        """Identifica o que gerou um áudio; se mudar (motor, amostra, parâmetros), o cache é invalidado."""
        if self.nome_voz == "reinaldo_haas":
            ref = pathlib.Path(self.ref_wav)
            return {
                "motor": "chatterbox-multilingual",
                "amostra": ref.name,
                "amostra_mtime": int(ref.stat().st_mtime),
                "cfg_weight": round(self.cfg_weight, 3),
                "exaggeration": round(self.exaggeration, 3),
                "temperature": CB_TEMPERATURE,
                "fp16": bool(self.usar_fp16),
            }
        if self.nome_voz == "reinaldo_f5":
            ref = pathlib.Path(self.ref_wav)
            return {
                "motor": "f5-tts-ptbr",
                "repo": F5_REPO,
                "amostra": ref.name,
                "amostra_mtime": int(ref.stat().st_mtime),
                "ref_text": self.ref_text,
                "nfe_step": F5_NFE_STEP,
                "cfg": F5_CFG,
                "speed": F5_SPEED,
            }
        return {"motor": f"edge-tts:{self.nome_voz}"}

    # ---------------- F5-TTS ----------------
    @staticmethod
    def _preparar_referencia_f5(ref_wav: pathlib.Path):
        """Recorta a referência (sem silêncio inicial, até F5_REF_MAX_S s) e obtém sua transcrição.
        A transcrição fica em um .txt ao lado — EDITE-O se houver erros: o F5 depende dela."""
        ref_wav = pathlib.Path(ref_wav)
        clip = ref_wav.with_name(ref_wav.stem + f"_f5ref{F5_REF_MAX_S}s.wav")
        txt = clip.with_suffix(".txt")
        if not clip.exists() or clip.stat().st_mtime < ref_wav.stat().st_mtime:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(ref_wav),
                 "-af", "silenceremove=start_periods=1:start_threshold=-40dB:start_silence=0.2",
                 "-t", str(F5_REF_MAX_S), "-ar", "24000", "-ac", "1", str(clip)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if txt.exists():
                txt.unlink()   # referência mudou -> transcrever de novo
            print(f"Referência F5 recortada: {clip.name}")
        if txt.exists() and txt.read_text(encoding="utf-8").strip():
            return clip, txt.read_text(encoding="utf-8").strip()
        print("Transcrevendo a referência (uma vez, faster-whisper 'small' na CPU)...")
        from faster_whisper import WhisperModel
        modelo = WhisperModel("small", device="cpu", compute_type="int8")
        segs, _ = modelo.transcribe(str(clip), language="pt", beam_size=5)
        texto = " ".join(s.text.strip() for s in segs).strip()
        del modelo
        txt.write_text(texto, encoding="utf-8")
        print(f"  Transcrição salva em {txt.name} — confira e corrija se necessário.")
        return clip, texto

    def _carregar_f5(self):
        """Localiza checkpoint e vocab do modelo pt-BR no cache do Hugging Face e carrega o F5-TTS."""
        from huggingface_hub import snapshot_download
        from f5_tts.api import F5TTS
        pasta = pathlib.Path(snapshot_download(F5_REPO, local_files_only=(os.environ.get("HF_HUB_OFFLINE") == "1")))
        ckpts = sorted(list(pasta.rglob("*.safetensors")) + list(pasta.rglob("*.pt")),
                       key=lambda f: f.stat().st_size, reverse=True)
        vocabs = list(pasta.rglob("vocab.txt"))
        if not ckpts:
            raise FileNotFoundError(f"Nenhum checkpoint (.safetensors/.pt) encontrado em {pasta}")
        ckpt = str(ckpts[0])
        vocab = str(vocabs[0]) if vocabs else ""
        print(f"Carregando F5-TTS pt-BR ({self.device}): {ckpts[0].name}" + (f" + {vocabs[0].name}" if vocabs else ""))
        erro = None
        for arq in F5_ARQUITETURAS:
            try:
                self.tts = F5TTS(model=arq, ckpt_file=ckpt, vocab_file=vocab, device=self.device)
                print(f"  Arquitetura: {arq}")
                return
            except Exception as e:
                erro = e
                print(f"  ({arq} não bateu: {str(e)[:90]}...)")
        raise RuntimeError(f"Não foi possível carregar o F5-TTS com nenhuma arquitetura conhecida: {erro}")

    def _gerar_frase_f5(self, frase: str):
        import numpy as np
        with torch.inference_mode():
            wav, sr, _ = self.tts.infer(
                ref_file=str(self.ref_wav), ref_text=self.ref_text, gen_text=frase,
                nfe_step=F5_NFE_STEP, cfg_strength=F5_CFG, speed=F5_SPEED,
                remove_silence=False, show_info=lambda *a, **k: None, progress=None,
            )
        self.sr = sr
        t = torch.from_numpy(np.asarray(wav, dtype=np.float32))
        return t.unsqueeze(0) if t.dim() == 1 else t

    @staticmethod
    def _preparar_referencia(amostra: pathlib.Path) -> pathlib.Path:
        """Converte a amostra para wav mono 24 kHz (uma vez, com cache)."""
        ref_wav = amostra.with_name(amostra.stem + "_ref24k.wav")
        if not ref_wav.exists() or ref_wav.stat().st_mtime < amostra.stat().st_mtime:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(amostra), "-ar", "24000", "-ac", "1",
                 "-af", "loudnorm=I=-20:TP=-2", str(ref_wav)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print(f"Referência preparada: {ref_wav.name}")
        return ref_wav

    # ---------------- clonagem local ----------------
    def _sintetizar_clonado(self, texto_fala: str, audio_out_wav: pathlib.Path):
        import torchaudio
        pedacos = []
        for frase in dividir_em_frases(texto_fala):
            wav = self._gerar_frase_f5(frase) if self.nome_voz == "reinaldo_f5" else self._gerar_frase(frase)
            pedacos.append(wav.cpu())
            pedacos.append(torch.zeros(1, int(PAUSA_ENTRE_FRASES * self.sr)))
        sr = self.sr
        if pedacos:
            pedacos.pop()  # remove o silêncio final
        audio = torch.cat(pedacos, dim=1)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        # Normaliza volume para consistência entre slides
        pico = audio.abs().max().clamp(min=1e-6)
        audio = audio / pico * 0.9
        torchaudio.save(str(audio_out_wav), audio, sr)
        return audio.shape[1] / sr

    def _gerar_frase(self, frase: str):
        """Gera uma frase; em GPU tenta precisão mista (fp16) para ganhar velocidade e VRAM,
        e cai para fp32 automaticamente se o modelo reclamar."""
        kwargs = dict(language_id="pt", exaggeration=self.exaggeration, cfg_weight=self.cfg_weight,
                      temperature=CB_TEMPERATURE)
        if not getattr(self, "conds_prontos", False):
            kwargs["audio_prompt_path"] = str(self.ref_wav)
        if self.device == "cuda" and self.usar_fp16:
            try:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    return self.tts.generate(frase, **kwargs)
            except Exception as e:
                print(f"  (fp16 falhou: {str(e)[:80]}... usando fp32 daqui em diante)")
                self.usar_fp16 = False
                torch.cuda.empty_cache()
        with torch.inference_mode():
            return self.tts.generate(frase, **kwargs)

    def _alinhar_palavras(self, texto_fala: str, audio_wav: pathlib.Path, total_dur: float) -> list:
        """Alinhamento forçado: o texto é conhecido, só recuperamos os tempos."""
        import whisperx
        audio = whisperx.load_audio(str(audio_wav))
        segmentos = [{"text": texto_fala, "start": 0.0, "end": total_dur}]
        resultado = whisperx.align(
            segmentos, self.align_model, self.align_meta, audio, self.align_device,
            return_char_alignments=False
        )
        palavras = []
        for w in resultado.get("word_segments", []):
            palavras.append((w.get("word", "").strip(), w.get("start"), w.get("end")))
        palavras = [p for p in palavras if p[0]]
        if not palavras:
            # Fallback: distribui as palavras uniformemente
            toks = texto_fala.split()
            passo = total_dur / max(len(toks), 1)
            return [(t, i * passo, (i + 1) * passo) for i, t in enumerate(toks)]
        return _preencher_tempos_faltantes(palavras, total_dur)

    # ---------------- fallback edge-tts ----------------
    async def _sintetizar_edge(self, texto_fala: str, audio_out_wav: pathlib.Path, work_dir: pathlib.Path):
        import edge_tts
        voice_base = "pt-BR-FranciscaNeural" if self.nome_voz == "francisca" else "pt-BR-AntonioNeural"
        c = edge_tts.Communicate(texto_fala, voice_base, boundary="WordBoundary")
        audio_data = bytearray()
        words_timing = []
        async for chunk in c.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 10_000_000.0
                dur = chunk["duration"] / 10_000_000.0
                words_timing.append((chunk["text"], start, start + dur))
        raw_mp3 = work_dir / "temp_tts_base.mp3"
        with open(raw_mp3, "wb") as f:
            f.write(audio_data)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_mp3), "-ar", "24000", "-ac", "1", str(audio_out_wav)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return words_timing

    # ---------------- interface usada pelo pipeline ----------------
    async def sintetizar_slide(self, texto_fala: str, audio_out_wav: pathlib.Path, work_dir: pathlib.Path):
        if self.nome_voz in VOZES_CLONADAS:
            t1 = time.time()
            total_dur = self._sintetizar_clonado(texto_fala, audio_out_wav)
            t2 = time.time()
            words_timing = self._alinhar_palavras(texto_fala, audio_out_wav, total_dur)
            print(f"  [tempos] síntese {t2-t1:.1f}s | alinhamento {time.time()-t2:.1f}s | áudio {total_dur:.1f}s")
            return total_dur, words_timing

        words_timing = await self._sintetizar_edge(texto_fala, audio_out_wav, work_dir)
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_out_wav)],
            capture_output=True, text=True, check=True
        )
        return float(res.stdout.strip()), words_timing


    async def obter_timing_rapido(self, texto_fala: str, audio_out_wav: pathlib.Path):
        """Recupera só os tempos por palavra de um áudio já sintetizado (cache), sem ressintetizar."""
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_out_wav)],
            capture_output=True, text=True, check=True
        )
        total_dur = float(res.stdout.strip())
        if self.nome_voz in VOZES_CLONADAS:
            return total_dur, self._alinhar_palavras(texto_fala, audio_out_wav, total_dur)
        # edge-tts: reconsulta apenas os WordBoundary (rápido, sem baixar áudio novo)
        import edge_tts
        voice_base = "pt-BR-FranciscaNeural" if self.nome_voz == "francisca" else "pt-BR-AntonioNeural"
        c = edge_tts.Communicate(texto_fala, voice_base, boundary="WordBoundary")
        words_timing = []
        async for chunk in c.stream():
            if chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 10_000_000.0
                dur = chunk["duration"] / 10_000_000.0
                words_timing.append((chunk["text"], start, start + dur))
        return total_dur, words_timing

# -------------------------------------------------------------
# Pipeline Principal
# -------------------------------------------------------------
async def executar(
    pptx_path: pathlib.Path,
    roteiro_path: pathlib.Path = None,
    nome_voz: str = "reinaldo_haas",
    amostra_ref: pathlib.Path = None,
    saida_video: pathlib.Path = None,
    slides_filtro: list = None,
    aviso_inicio: str = None,
    aviso_fim: str = None,
    duracao_aviso: float = 5.0,
    cfg_weight: float = CB_CFG_WEIGHT,
    exaggeration: float = CB_EXAGGERATION,
    permitir_cpu: bool = False,
    usar_fp16: bool = False,
    offset_sincronia: float = -0.12,
    forcar: bool = False,
    so_clipes: bool = False,          # trabalhador paralelo: gera clipes e não concatena
    banner_primeiro: int = None,      # nº global do primeiro/último slide (para o aviso de topo nos trabalhadores)
    banner_ultimo: int = None
):
    work_dir = pptx_path.resolve().parent / "video_work" / pptx_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    slides_img_dir = work_dir / "slides_img"
    audio_dir = work_dir / "audio"
    subs_dir = work_dir / "legendas"
    clips_dir = work_dir / "clipes"

    audio_dir.mkdir(parents=True, exist_ok=True)
    subs_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    if roteiro_path and roteiro_path.exists():
        print(f"Usando roteiro customizado: {roteiro_path}")
        with open(roteiro_path, "r", encoding="utf-8-sig") as f:
            roteiro = json.load(f)
        if slides_filtro:
            roteiro = [it for it in roteiro if it["slide"] in slides_filtro]
        # Exporta apenas as imagens que faltam, e só dos slides que serão renderizados
        faltantes = [it["slide"] for it in roteiro
                     if not (slides_img_dir / f"slide_{it['slide']:03d}.png").exists()]
        if faltantes:
            print(f"Exportando imagens faltantes dos slides: {faltantes}")
            extrair_e_montar_roteiro(pptx_path, slides_img_dir, slides_filtro=faltantes)
    else:
        roteiro = extrair_e_montar_roteiro(pptx_path, slides_img_dir, slides_filtro=slides_filtro)

    if not roteiro:
        print("Aviso: Nenhum slide selecionado para renderização.")
        return

    primeiro_slide_num = roteiro[0]["slide"]
    ultimo_slide_num = roteiro[-1]["slide"]
    slide_aviso_inicio = banner_primeiro if banner_primeiro is not None else primeiro_slide_num
    slide_aviso_fim = banner_ultimo if banner_ultimo is not None else ultimo_slide_num

    gpu = torch.cuda.is_available()
    print(f"CPUs utilizáveis: {_N_CPU} (threads torch: {torch.get_num_threads()})")
    if gpu:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb < 6:
            print(f"Aviso: GPU com {vram_gb:.1f} GB de VRAM. Se a sintese ficar lenta (< 10 it/s), a memoria esta "
                  f"estourando para a RAM. Feche outros programas que usem a GPU (navegador, PowerPoint com aceleracao).")
    print(f"\nIniciando sintese e renderizacao ({len(roteiro)} slides: {primeiro_slide_num} a {ultimo_slide_num}, "
          f"voz '{nome_voz}', dispositivo: {'GPU ' + torch.cuda.get_device_name(0) if gpu else 'CPU'})...")
    try:
        motor = MotorVoz(nome_voz, amostra_ref=amostra_ref, cfg_weight=cfg_weight,
                         exaggeration=exaggeration, permitir_cpu=permitir_cpu, usar_fp16=usar_fp16)
    except RuntimeError as e:
        if "GPU nao detectada" in str(e):
            print(f"\nERRO: {e}")
            return
        raise
    except Exception as e:
        msg = str(e)
        if os.environ.get("HF_HUB_OFFLINE") == "1" and any(
            k in msg for k in ("offline", "Offline", "cache", "Cannot find", "not found", "LocalEntryNotFound", "connection")
        ):
            print("\nERRO: um modelo não está no cache local e o script está em modo offline.")
            print("Rode UMA vez com --online para baixar os modelos; nas próximas execuções não precisa mais.")
            print(f"Detalhe: {msg[:300]}")
            return
        raise

    clipes_gerados = []
    for item in roteiro:
        s_num = item["slide"]
        texto_fala = item.get("texto_fala", "").strip()
        texto_original = item.get("texto_original", texto_fala).strip()

        if not texto_fala:
            continue

        print(f"\n[Slide {s_num:03d}]")
        print(f"  Anotações do PPT: {texto_fala[:90]}...")

        audio_wav = audio_dir / f"slide_{s_num:03d}.wav"
        ass_path = subs_dir / f"slide_{s_num:03d}.ass"
        words_json = subs_dir / f"slide_{s_num:03d}_words.json"
        clip_mp4 = clips_dir / f"clip_{s_num:03d}.mp4"
        img_path = slides_img_dir / f"slide_{s_num:03d}.png"

        # Cache seguro: o áudio só é reaproveitado se foi gerado com o MESMO texto,
        # MESMO motor de voz, MESMA amostra e MESMOS parâmetros. Qualquer diferença regenera.
        # Áudios sem metadados (gerados por versões antigas do script) são sempre regenerados.
        assinatura_atual = motor.assinatura()
        dados_cache = None
        motivo = None
        if not audio_wav.exists() or audio_wav.stat().st_size <= 1000:
            motivo = "sem áudio"
        elif not words_json.exists():
            motivo = "áudio sem metadados (versão antiga do script)"
        else:
            try:
                with open(words_json, "r", encoding="utf-8") as f:
                    dados_cache = json.load(f)
                if dados_cache.get("texto_fala") != texto_fala:
                    motivo = "texto das anotações alterado no PPTX"
                elif dados_cache.get("assinatura") != assinatura_atual:
                    motivo = "motor de voz, amostra ou parâmetros diferentes"
            except Exception:
                motivo = "metadados ilegíveis"
        if forcar:
            motivo = "--forcar"

        t0 = time.time()
        if motivo is None:
            dur = dados_cache["dur"]
            words_timing = dados_cache["words_timing"]
            print(f"  Áudio em cache ({dur:.2f}s). Atualizando legendas e clipe...")
        else:
            print(f"  Sintetizando áudio ({motivo})...")
            dur, words_timing = await motor.sintetizar_slide(texto_fala, audio_wav, work_dir)
            with open(words_json, "w", encoding="utf-8") as f:
                json.dump({"dur": dur, "texto_fala": texto_fala, "words_timing": words_timing,
                           "assinatura": assinatura_atual}, f, ensure_ascii=False)
            print(f"  Áudio sintetizado: {dur:.2f}s (processado em {time.time()-t0:.2f}s)")

        banner_topo = None
        if s_num == slide_aviso_inicio and aviso_inicio:
            banner_topo = aviso_inicio
        elif s_num == slide_aviso_fim and aviso_fim:
            banner_topo = aviso_fim

        ass_content = gerar_ass_wordboundary(
            words_timing=words_timing,
            texto_original=texto_original,
            chunk_size=7,
            aviso_topo=banner_topo,
            duracao_aviso=duracao_aviso,
            total_dur=dur,
            offset_sincronia=offset_sincronia
        )
        ass_path.write_text(ass_content, encoding="utf-8")

        clean_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
        # Imagem parada + legendas: 10 fps bastam (destaque de palavras com resolução de 0,1 s) e cortam
        # o custo do x264 em ~2,5x; preset veryfast + threads limitadas para conviver com trabalhadores paralelos.
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-loop", "1", "-framerate", str(FPS_CLIPE),
            "-i", str(img_path),
            "-i", str(audio_wav),
            "-vf", f"subtitles='{clean_ass}'",
            "-r", str(FPS_CLIPE),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-tune", "stillimage",
            "-threads", str(FFMPEG_THREADS),
            "-c:a", "aac", "-b:a", "160k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(clip_mp4)
        ]
        t_ff = time.time()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  Clipe renderizado em {time.time()-t_ff:.1f}s")
        clipes_gerados.append(clip_mp4)

    if not clipes_gerados:
        print("Erro: Nenhum clipe gerado.")
        return

    if so_clipes:
        print(f"\n[trabalhador] {len(clipes_gerados)} clipe(s) prontos: slides {primeiro_slide_num}-{ultimo_slide_num}")
        return

    concatenar_clipes(pptx_path, work_dir, clipes_gerados, saida_video, slides_filtro, primeiro_slide_num, ultimo_slide_num)


def concatenar_clipes(pptx_path, work_dir, clipes_gerados, saida_video, slides_filtro, primeiro_slide_num, ultimo_slide_num):
    concat_txt = work_dir / "concat_list.txt"
    with open(concat_txt, "w", encoding="utf-8") as f:
        for c in clipes_gerados:
            clean_p = str(c.resolve()).replace("\\", "/")
            f.write(f"file '{clean_p}'\n")

    # Salva sempre na pasta 'video'
    if not saida_video:
        pasta_destino = pptx_path.resolve().parent / "video"
        if not pasta_destino.exists():
            pasta_destino = pptx_path.resolve().parent / "videos"
        pasta_destino.mkdir(parents=True, exist_ok=True)

        if len(clipes_gerados) == 1 and slides_filtro and len(slides_filtro) == 1:
            saida_video = pasta_destino / f"{pptx_path.stem}_slide_{primeiro_slide_num:03d}.mp4"
        elif slides_filtro:
            saida_video = pasta_destino / f"{pptx_path.stem}_slides_{primeiro_slide_num}_a_{ultimo_slide_num}.mp4"
        else:
            saida_video = pasta_destino / f"{pptx_path.stem}.mp4"

    print(f"\nConcatenando todos os clipes em:\n{saida_video}...")
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_txt),
        "-c", "copy",
        str(saida_video)
    ]
    subprocess.run(cmd_concat, check=True)
    print("\n=======================================================")
    print("VÍDEO DA AULA GERADO COM SUCESSO:")
    print(f"Arquivo: {saida_video}")
    print(f"Tamanho: {saida_video.stat().st_size} bytes")
    print("=======================================================")

# -------------------------------------------------------------
# Escolha automática da GPU em nó compartilhado
# -------------------------------------------------------------
def quadro_gpus(leituras: int = 2, intervalo: float = 3.0) -> dict:
    """{indice: (gb_livres_min, util_max)} ao longo de `leituras` consultas ao nvidia-smi."""
    def ler():
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return {int(i): (float(f) / 1024, int(u)) for i, f, u in (l.split(",") for l in out.strip().splitlines())}
    lidas = [ler()]
    for _ in range(leituras - 1):
        time.sleep(intervalo); lidas.append(ler())
    return {i: (min(l[i][0] for l in lidas), max(l[i][1] for l in lidas)) for i in lidas[0]}


def escolher_gpu(pedido: str = "auto", min_livre_gb: float = 8.0, leituras: int = 2, intervalo: float = 3.0):
    """Define CUDA_VISIBLE_DEVICES antes de qualquer uso de CUDA.
    pedido: 'auto' (placa mais ociosa com memória livre), um índice ('3'), ou 'todas' (não mexe)."""
    if pedido == "todas":
        return
    if pedido != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = pedido
        print(f"GPU fixada: {pedido}")
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"GPU (do ambiente): {os.environ['CUDA_VISIBLE_DEVICES']}")
        return
    try:
        def ler():
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
            return {int(i): (float(f) / 1024, int(u)) for i, f, u in (l.split(",") for l in out.strip().splitlines())}
        lidas = [ler()]
        if len(lidas[0]) <= 1:
            return                                   # uma GPU só: nada a escolher
        for _ in range(leituras - 1):
            time.sleep(intervalo); lidas.append(ler())
        placas = []
        for i in lidas[0]:
            livre = min(l[i][0] for l in lidas); util = max(l[i][1] for l in lidas)
            placas.append((util, -livre, i, livre))
        placas.sort()
        print("GPUs: " + "  ".join(f"[{i}] {livre:.0f}GB livres/{util}%" for util, _, i, livre in sorted(placas, key=lambda x: x[2])))
        candidatas = [p for p in placas if p[3] >= min_livre_gb]
        if not candidatas:
            print(f"Aviso: nenhuma GPU com >= {min_livre_gb:.0f} GB livres; usando a mais ociosa mesmo assim.")
            candidatas = placas
        util, _, idx, livre = candidatas[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        print(f"GPU escolhida: {idx} ({livre:.0f} GB livres, {util}% de uso)")
    except Exception as e:
        print(f"(não foi possível consultar nvidia-smi: {str(e)[:60]}; usando a GPU padrão)")


# -------------------------------------------------------------
# Modo paralelo: N trabalhadores na mesma GPU, cada um com uma fatia dos slides
# -------------------------------------------------------------
def executar_paralelo(args, pptx_path: pathlib.Path, slides_filtro: list, n: int):
    """O Chatterbox gera um token por vez e usa só uma fração de uma GPU grande; N processos
    em paralelo dão speedup quase linear até saturar a placa (~3,5 GB de VRAM por trabalhador)."""
    work_dir = pptx_path.resolve().parent / "video_work" / pptx_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    slides_img_dir = work_dir / "slides_img"

    # 0) Trava: não iniciar se já há trabalhadores deste PPTX rodando (disputariam a GPU e estourariam a VRAM)
    if sys.platform != "win32":
        try:
            ps = subprocess.run(["pgrep", "-f", f"gerador_de_video_e_som.py.*{re.escape(pptx_path.name)}.*--so-clipes"],
                                capture_output=True, text=True).stdout.split()
            ps = [q for q in ps if int(q) != os.getpid()]
            if ps:
                print(f"ERRO: já existem {len(ps)} trabalhador(es) desta apresentação em execução (PIDs {' '.join(ps)}).")
                print("  Acompanhe no terminal original ou encerre com:  pkill -f gerador_de_video_e_som.py")
                return
        except Exception:
            pass

    # 0b) Plano de GPUs. Sem o serviço MPS da NVIDIA, vários processos numa MESMA placa se revezam a cada
    # kernel e um modelo autorregressivo quase não ganha com isso; o ganho real vem de usar VÁRIAS placas.
    VRAM_POR_TRABALHADOR_GB = 7.0
    MAX_POR_GPU = args.por_gpu
    slots = []                                        # lista de índices de GPU, um por trabalhador
    fixo = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.gpu not in ("auto", "todas"):
        fixo = args.gpu
    try:
        quadro = quadro_gpus()
    except Exception:
        quadro = {}
    if fixo:
        idx = [int(x) for x in fixo.split(",") if x.strip().isdigit()]
        for i in idx:
            livre = quadro.get(i, (999, 0))[0]
            slots += [i] * max(1, min(MAX_POR_GPU, int(livre / VRAM_POR_TRABALHADOR_GB)))
    elif quadro:
        print("GPUs: " + "  ".join(f"[{i}] {g[0]:.0f}GB livres/{g[1]}%" for i, g in sorted(quadro.items())))
        ociosas = [(u, -l, i) for i, (l, u) in quadro.items() if u <= args.gpu_util_max and l >= VRAM_POR_TRABALHADOR_GB]
        for _, _, i in sorted(ociosas):
            livre = quadro[i][0]
            slots += [i] * min(MAX_POR_GPU, int(livre / VRAM_POR_TRABALHADOR_GB))
        if not slots:
            print("Nenhuma GPU ociosa com memória livre; usando a menos carregada com 1 trabalhador.")
            i = sorted((u, -l, i) for i, (l, u) in quadro.items())[0][2]
            slots = [i]
    if not slots:
        slots = [None]                                # sem nvidia-smi: deixa o ambiente decidir
    # Intercala as placas para que os primeiros N slots cubram o máximo de GPUs distintas
    por_gpu = {}
    for s in slots:
        por_gpu.setdefault(s, []).append(s)
    intercalado = []
    while any(por_gpu.values()):
        for g in list(por_gpu):
            if por_gpu[g]:
                intercalado.append(por_gpu[g].pop())
    slots = intercalado
    if n > len(slots):
        print(f"Aviso: --paralelo {n} -> {len(slots)} trabalhador(es), limitado por GPUs ociosas/VRAM "
              f"(máx. {MAX_POR_GPU} por placa; ajuste com --por-gpu).")
        n = len(slots)
    slots = slots[:n]

    # 1) Roteiro e imagens, uma única vez, no processo pai (evita corrida no LibreOffice/PowerPoint)
    if args.roteiro and pathlib.Path(args.roteiro).exists():
        roteiro_path = pathlib.Path(args.roteiro).resolve()
        with open(roteiro_path, "r", encoding="utf-8-sig") as f:
            roteiro = json.load(f)
        faltantes = [it["slide"] for it in roteiro
                     if (not slides_filtro or it["slide"] in slides_filtro)
                     and not (slides_img_dir / f"slide_{it['slide']:03d}.png").exists()]
        if faltantes:
            extrair_e_montar_roteiro(pptx_path, slides_img_dir, slides_filtro=faltantes)
    else:
        roteiro = extrair_e_montar_roteiro(pptx_path, slides_img_dir, slides_filtro=slides_filtro)
        roteiro_path = work_dir / "roteiro_gerado_automaticamente.json"
        with open(roteiro_path, "w", encoding="utf-8") as f:
            json.dump(roteiro, f, indent=2, ensure_ascii=False)
    if slides_filtro:
        roteiro = [it for it in roteiro if it["slide"] in slides_filtro]
    roteiro = [it for it in roteiro if it.get("texto_fala", "").strip()]
    if not roteiro:
        print("Aviso: nenhum slide com texto para sintetizar."); return
    primeiro, ultimo = roteiro[0]["slide"], roteiro[-1]["slide"]

    # 2) Fatias balanceadas pelo tamanho do texto (round-robin sobre slides ordenados por tamanho)
    n = max(1, min(n, len(roteiro)))
    por_tamanho = sorted(roteiro, key=lambda it: -len(it["texto_fala"]))
    fatias = [[] for _ in range(n)]
    carga = [0] * n
    for it in por_tamanho:
        k = carga.index(min(carga))
        fatias[k].append(it["slide"]); carga[k] += len(it["texto_fala"])
    fatias = [sorted(f) for f in fatias if f]

    # 3) Lança os trabalhadores
    base = [sys.executable, str(pathlib.Path(__file__).resolve()), str(pptx_path), "--roteiro", str(roteiro_path),
            "--so-clipes", "--gpu", "todas", "--banner-primeiro", str(primeiro), "--banner-ultimo", str(ultimo),
            "-v", args.voz, "--cfg", str(args.cfg), "--exag", str(args.exag), "--offset", str(args.offset),
            "--duracao-aviso", str(args.duracao_aviso), "--aviso-inicio", args.aviso_inicio, "--aviso-fim", args.aviso_fim]
    if args.amostra: base += ["-a", args.amostra]
    for flag in ("forcar", "fp16", "permitir_cpu", "online", "sem_aviso_inicio", "sem_aviso_fim"):
        if getattr(args, flag): base.append("--" + flag.replace("_", "-"))
    if args.hf_home: base += ["--hf-home", args.hf_home]

    logs_dir = work_dir / "logs"; logs_dir.mkdir(exist_ok=True)
    procs = []
    uso = {}
    for g in slots: uso[g] = uso.get(g, 0) + 1
    print(f"\nModo paralelo: {len(fatias)} trabalhadores, {len(roteiro)} slides; GPUs: "
          + ", ".join(f"{g} x{c}" for g, c in sorted(uso.items(), key=lambda x: str(x[0]))))
    t0 = time.time()
    for k, fatia in enumerate(fatias):
        log = logs_dir / f"trabalhador_{k+1:02d}.log"
        cmd = base + ["-s", ",".join(map(str, fatia))]
        env = os.environ.copy()
        if slots[k] is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(slots[k])
        print(f"  [{k+1:02d}] GPU {slots[k] if slots[k] is not None else '-'}  slides {fatia[0]}..{fatia[-1]} ({len(fatia)})  ->  {log.name}")
        procs.append((k + 1, fatia, subprocess.Popen(cmd, stdout=open(log, "w", encoding="utf-8"),
                                                     stderr=subprocess.STDOUT, env=env)))
        time.sleep(3)   # escalona a carga dos modelos

    # 4) Acompanha o progresso pelos clipes prontos
    clips_dir = work_dir / "clipes"
    esperados = {it["slide"] for it in roteiro}
    ultimo_print = 0
    while any(p.poll() is None for _, _, p in procs):
        time.sleep(10)
        prontos = sum(1 for s in esperados if (clips_dir / f"clip_{s:03d}.mp4").exists())
        if prontos != ultimo_print:
            print(f"  progresso: {prontos}/{len(esperados)} clipes  ({time.time()-t0:.0f}s)"); ultimo_print = prontos
    falhas = [(k, p.returncode) for k, _, p in procs if p.returncode != 0]
    if falhas:
        for k, rc in falhas:
            print(f"ERRO: trabalhador {k:02d} terminou com código {rc}. Veja {logs_dir / f'trabalhador_{k:02d}.log'}")
            print("  últimas linhas:"); print("   " + "\n   ".join(
                (logs_dir / f"trabalhador_{k:02d}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-8:]))
        print("Corrija e rode de novo — os slides já concluídos ficam em cache.")
        return

    # 5) Concatena na ordem original dos slides
    clipes = [clips_dir / f"clip_{it['slide']:03d}.mp4" for it in roteiro]
    faltam = [c.name for c in clipes if not c.exists()]
    if faltam:
        print(f"ERRO: clipes não gerados: {faltam}"); return
    print(f"\nSíntese concluída em {time.time()-t0:.0f}s. Concatenando...")
    concatenar_clipes(pptx_path, work_dir, clipes, pathlib.Path(args.saida) if args.saida else None,
                      slides_filtro, primeiro, ultimo)


# -------------------------------------------------------------
# CLI Entrypoint
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Gerador de Vídeo e Som Automatizado para Apresentações")
    parser.add_argument("pptx", nargs="?", help="Caminho do arquivo .pptx")
    parser.add_argument("--roteiro", "-r", help="Arquivo JSON de roteiro (opcional)")
    parser.add_argument("--voz", "-v", default="reinaldo_haas", choices=["reinaldo_haas", "reinaldo_f5", "antonio", "francisca"],
                        help="Voz da narração: reinaldo_haas = Chatterbox; reinaldo_f5 = F5-TTS pt-BR (mais leve e rápido); "
                             "antonio/francisca = edge-tts (nuvem). Padrao: reinaldo_haas")
    parser.add_argument("--amostra", "-a", help="Áudio de referência da voz clonada (padrão: voz_referencia/amostra_nova.mp3)")
    parser.add_argument("--cfg", type=float, default=CB_CFG_WEIGHT,
                        help=f"Chatterbox cfg_weight: menor = mais fiel ao timbre da amostra (padrao: {CB_CFG_WEIGHT})")
    parser.add_argument("--exag", type=float, default=CB_EXAGGERATION,
                        help=f"Chatterbox exaggeration: expressividade, 0.5 e neutro (padrao: {CB_EXAGGERATION})")
    parser.add_argument("--saida", "-o", help="Caminho do arquivo .mp4 de saída")
    parser.add_argument("--slides", "-s", help="Filtro de slides (ex: '1', '1-5', '1,3,7', '10-')")
    parser.add_argument("--aviso-inicio", default=DEFAULT_AVISO_INICIO,
                        help="Texto do aviso de topo no primeiro slide (suporta \\N ou quebra de linha)")
    parser.add_argument("--aviso-fim", default=DEFAULT_AVISO_FIM,
                        help="Texto do aviso de topo no último slide (suporta \\N ou quebra de linha)")
    parser.add_argument("--duracao-aviso", type=float, default=5.0,
                        help="Duração em segundos do aviso de topo (padrao: 5.0s)")
    parser.add_argument("--sem-aviso-inicio", action="store_true", help="Desativa o aviso de topo no primeiro slide")
    parser.add_argument("--sem-aviso-fim", action="store_true", help="Desativa o aviso de topo no último slide")
    parser.add_argument("--offset", type=float, default=-0.12,
                        help="Offset de sincronia das legendas em segundos (padrao: -0.12s para eliminar atraso perceptual)")
    parser.add_argument("--forcar", action="store_true", help="Força a regeração de todos os clipes e áudios do zero")
    parser.add_argument("--fp16", action="store_true",
                        help="Precisão mista na GPU: mais rápido e menos VRAM (recomendado em placas de 4-6 GB). "
                             "Se der erro, o script volta para fp32 sozinho.")
    parser.add_argument("--gpu", default="auto", metavar="N|auto|todas",
                        help="Qual GPU usar num nó com várias: 'auto' escolhe a mais ociosa com memória livre (padrão); "
                             "um índice fixa; 'todas' não altera CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--por-gpu", type=int, default=2, metavar="K",
                        help="No modo paralelo, máximo de trabalhadores por placa (padrão 2; sem MPS, mais que isso rende pouco)")
    parser.add_argument("--gpu-util-max", type=int, default=15, metavar="PCT",
                        help="No modo paralelo, só usa placas com utilização <= PCT%% (padrão 15)")
    parser.add_argument("--paralelo", "-p", type=int, default=1, metavar="N",
                        help="Sintetiza com N processos em paralelo na mesma GPU (~3,5 GB de VRAM cada). "
                             "Em GPU grande (>= 24 GB) use 4-8; speedup quase linear.")
    parser.add_argument("--so-clipes", action="store_true", help=argparse.SUPPRESS)       # uso interno (trabalhador)
    parser.add_argument("--banner-primeiro", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--banner-ultimo", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--permitir-cpu", action="store_true",
                        help="Permite sintetizar a voz clonada na CPU (muito lento, ~10 min por slide)")
    parser.add_argument("--online", action="store_true",
                        help="Permite ao Hugging Face consultar/baixar modelos pela internet. Use só na PRIMEIRA execução "
                             "(ou após limpar o cache); depois o padrão é 100%% offline, sem esperas de rede.")
    parser.add_argument("--hf-home", help="Pasta local do cache de modelos (padrão: %%USERPROFILE%%\\.cache\\huggingface). "
                                         "Evite pastas dentro do OneDrive.")

    args = parser.parse_args()

    # ---- Modelos: cache local e modo offline (deve ocorrer ANTES de importar chatterbox/whisperx) ----
    if args.hf_home:
        os.environ["HF_HOME"] = str(pathlib.Path(args.hf_home).resolve())
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.online:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        print("Modo ONLINE: modelos serão verificados/baixados do Hugging Face nesta execução.")
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if not args.so_clipes and args.paralelo <= 1:     # no modo paralelo o pai planeja as GPUs por trabalhador
        escolher_gpu(args.gpu)

    pptx_path = None
    if args.pptx:
        pptx_path = pathlib.Path(args.pptx)
    elif not sys.stdin.isatty():
        linha = sys.stdin.readline().strip()
        if linha:
            pptx_path = pathlib.Path(linha)

    if not pptx_path:
        candidatos = list(pathlib.Path(".").glob("*.pptx"))
        if candidatos:
            pptx_path = candidatos[0]
            print(f"Usando arquivo encontrado: {pptx_path.name}")
        else:
            print("Erro: Nenhum arquivo .pptx informado.")
            sys.exit(1)

    if not pptx_path.exists():
        print(f"Erro: Arquivo '{pptx_path}' nao encontrado.")
        sys.exit(1)

    slides_filtro = None
    if args.slides:
        slides_set = set()
        for parte in args.slides.split(","):
            parte = parte.strip()
            if not parte:
                continue
            if "-" in parte:
                pedacos = parte.split("-")
                if len(pedacos) == 2:
                    ini = int(pedacos[0]) if pedacos[0] else 1
                    fim = int(pedacos[1]) if pedacos[1] else 9999
                    slides_set.update(range(ini, fim + 1))
            else:
                slides_set.add(int(parte))
        slides_filtro = sorted(list(slides_set))

    aviso_ini = None if args.sem_aviso_inicio else args.aviso_inicio
    aviso_end = None if args.sem_aviso_fim else args.aviso_fim

    if args.paralelo > 1 and not args.so_clipes:
        executar_paralelo(args, pptx_path, slides_filtro, args.paralelo)
        return

    asyncio.run(executar(
        pptx_path=pptx_path,
        roteiro_path=pathlib.Path(args.roteiro) if args.roteiro else None,
        nome_voz=args.voz,
        amostra_ref=pathlib.Path(args.amostra) if args.amostra else None,
        saida_video=pathlib.Path(args.saida) if args.saida else None,
        slides_filtro=slides_filtro,
        aviso_inicio=aviso_ini,
        aviso_fim=aviso_end,
        duracao_aviso=args.duracao_aviso,
        cfg_weight=args.cfg,
        exaggeration=args.exag,
        permitir_cpu=args.permitir_cpu,
        usar_fp16=args.fp16,
        offset_sincronia=args.offset,
        forcar=args.forcar,
        so_clipes=args.so_clipes,
        banner_primeiro=args.banner_primeiro,
        banner_ultimo=args.banner_ultimo
    ))

if __name__ == "__main__":
    main()
