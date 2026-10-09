"""
gerador_de_video_e_som.py — vídeo-aula a partir de um .pptx com a voz clonada do professor.

Pipeline em quatro estágios, cada um uma função pura sobre arquivos:

  1. extrair      .pptx  ->  slides PNG + roteiro (texto das anotações do orador)
  2. sintetizar   texto  ->  .wav com a voz clonada  +  tempo de cada palavra
  3. legendar     tempos ->  .ass (palavra atual em destaque, pontuação original)
  4. renderizar   PNG + wav + ass  ->  clipe .mp4 por slide  ->  concatenação

Motores de voz (--voz):
  reinaldo_haas  Chatterbox Multilingual, clonagem zero-shot (padrão)
  reinaldo_f5    F5-TTS ajustado em pt-BR (mais leve e rápido)
  antonio / francisca   edge-tts (nuvem Microsoft), sem clonagem

Cache: cada áudio guarda uma assinatura (texto, motor, amostra, parâmetros). Só regenera o que mudou.
Tudo roda localmente; o Hugging Face só é consultado com --online.

Uso:
  python gerador_de_video_e_som.py aula.pptx                 # aula inteira
  python gerador_de_video_e_som.py aula.pptx -s 3-5          # alguns slides
  python gerador_de_video_e_som.py aula.pptx --paralelo 6    # várias GPUs
  python gerador_de_video_e_som.py -h
"""
import argparse, asyncio, json, os, pathlib, re, shutil, subprocess, sys, tempfile, time

# ------------------------------------------------------------------ CPU threads (antes do torch)
def _cpus() -> int:
    n = os.cpu_count() or 1
    try: n = min(n, len(os.sched_getaffinity(0)))
    except Exception: pass
    try:
        q = open("/sys/fs/cgroup/cpu.max").read().split()
        if q[0] != "max": n = min(n, max(1, int(q[0]) // int(q[1])))
    except Exception: pass
    return n
N_CPU = _cpus()
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, str(min(N_CPU, 16)))
import torch
torch.set_num_threads(min(N_CPU, 16))

# ------------------------------------------------------------------ configuração
AQUI = pathlib.Path(__file__).resolve().parent
AMOSTRA_PADRAO = AQUI / "voz_referencia" / "amostra_nova.mp3"
WINDOWS = sys.platform == "win32"
FONTE_LEGENDA = "Segoe UI" if WINDOWS else "DejaVu Sans"
VOZES_CLONADAS = ("reinaldo_haas", "reinaldo_f5")

CB_CFG, CB_EXAG, CB_TEMP = 0.4, 0.45, 0.7       # Chatterbox: cfg menor = mais fiel ao timbre
F5_REPO = "traderpedroso/F5-TTS-BRAZILIAN-PORTUGUESE"
F5_NFE, F5_CFG, F5_SPEED, F5_REF_S = 32, 2.0, 0.95, 12
PAUSA_FRASES = 0.35                              # s de silêncio entre frases
FPS_CLIPE, FFMPEG_THREADS = 10, 4                # imagem parada: 10 fps bastam para as legendas

DEFAULT_AVISO_INICIO = (r"{\c&H00D7FF&\b1}UFSC · PROF. REINALDO HAAS\N"
                        r"{\c&HFFFFFF&\b1}VOZ CLONADA LOCALMENTE POR PROGRAMA DE COMPUTADOR")
DEFAULT_AVISO_FIM = (r"{\c&H00D7FF&\b1}UFSC · PROF. REINALDO HAAS\N"
                     r"{\c&HFFFFFF&\b1}FIM DA AULA · BONS ESTUDOS COM OS EXERCÍCIOS!")


def tmp_local(nome: str) -> pathlib.Path:
    """Arquivo em disco local: o /home de um cluster (NFS/FUSE) é lento para leituras repetidas."""
    d = pathlib.Path(tempfile.gettempdir()) / f"videoaula_{os.getpid()}"
    d.mkdir(exist_ok=True)
    return d / nome


def duracao(wav: pathlib.Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(wav)], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


# ================================================================== 1. EXTRAIR
def normalizar_para_fala(texto: str) -> str:
    t = texto
    # Número de página solto no fim das anotações (mas não "Capítulo 3", tratado antes)
    t = re.sub(r'\b(Cap[ií]tulo)\s+(\d+)\s*$', lambda m: f"{m.group(1)} {m.group(2)}.", t, flags=re.IGNORECASE)
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



def _item(i, img, notas, textos_slide):
    base = notas.strip() or " ".join(textos_slide)
    fala = normalizar_para_fala(base)
    return {"slide": i, "imagem": img.name, "texto_fala": fala, "texto_original": base}


def extrair(pptx: pathlib.Path, img_dir: pathlib.Path, slides: list = None) -> list:
    """Exporta os slides pedidos para PNG 1920x1080 (com cache) e lê as anotações do orador."""
    img_dir.mkdir(parents=True, exist_ok=True)
    return (_extrair_windows if WINDOWS else _extrair_linux)(pptx, img_dir, slides)


def _precisa_png(img: pathlib.Path, pptx: pathlib.Path) -> bool:
    return not img.exists() or img.stat().st_size == 0 or img.stat().st_mtime < pptx.stat().st_mtime


def _extrair_windows(pptx, img_dir, slides):
    import win32com.client
    app = win32com.client.Dispatch("PowerPoint.Application")
    roteiro = []
    try:
        pres = app.Presentations.Open(str(pptx.resolve()), ReadOnly=True, Untitled=False, WithWindow=False)
        alvos = [s for s in range(1, pres.Slides.Count + 1) if not slides or s in slides]
        print(f"{pptx.name}: {pres.Slides.Count} slides, preparando {len(alvos)} [PowerPoint]")
        for i in alvos:
            sl = pres.Slides(i); img = img_dir / f"slide_{i:03d}.png"
            if _precisa_png(img, pptx):
                sl.Export(str(img), "PNG", 1920, 1080)
            notas = ""
            if sl.HasNotesPage:
                for sh in sl.NotesPage.Shapes:
                    if sh.HasTextFrame and sh.TextFrame.HasText:
                        t = sh.TextFrame.TextRange.Text.strip()
                        if t and not t.isdigit(): notas += " " + t
            textos = [sh.TextFrame.TextRange.Text.strip() for sh in sl.Shapes
                      if sh.HasTextFrame and sh.TextFrame.HasText and sh.TextFrame.TextRange.Text.strip()]
            roteiro.append(_item(i, img, notas, textos))
        pres.Close()
    finally:
        try: app.Quit()
        except Exception: pass
    return roteiro


def _extrair_linux(pptx, img_dir, slides):
    """LibreOffice headless -> PDF -> pdftoppm; anotações via python-pptx."""
    from pptx import Presentation
    pres = Presentation(str(pptx))
    total = len(pres.slides)
    alvos = [s for s in range(1, total + 1) if not slides or s in slides]
    print(f"{pptx.name}: {total} slides, preparando {len(alvos)} [LibreOffice]")
    faltam = [i for i in alvos if _precisa_png(img_dir / f"slide_{i:03d}.png", pptx)]
    if faltam:
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if not soffice or not shutil.which("pdftoppm"):
            raise RuntimeError("Preciso de LibreOffice (soffice) e poppler (pdftoppm): "
                               "mamba install -c conda-forge libreoffice poppler")
        pdf = img_dir / (pptx.stem + ".pdf")
        if not pdf.exists() or pdf.stat().st_mtime < pptx.stat().st_mtime:
            subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(img_dir), str(pptx)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        for i in faltam:
            pref = img_dir / f"_p{i:03d}"
            subprocess.run(["pdftoppm", "-png", "-f", str(i), "-l", str(i), "-scale-to-x", "1920",
                            "-scale-to-y", "1080", str(pdf), str(pref)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            gerados = sorted(img_dir.glob(f"_p{i:03d}*.png"))
            gerados[0].replace(img_dir / f"slide_{i:03d}.png")
            for g in gerados[1:]: g.unlink()
    roteiro = []
    for i in alvos:
        sl = pres.slides[i - 1]
        notas = sl.notes_slide.notes_text_frame.text if sl.has_notes_slide and sl.notes_slide.notes_text_frame else ""
        textos = [sh.text_frame.text.strip() for sh in sl.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
        roteiro.append(_item(i, img_dir / f"slide_{i:03d}.png", notas, textos))
    return roteiro


# ================================================================== 2. SINTETIZAR
def dividir_em_frases(texto: str, max_chars: int = 220) -> list:
    """Clonagem zero-shot rende melhor em trechos curtos."""
    frases = []
    for p in re.split(r"(?<=[.!?;:])\s+", texto.strip()):
        p = p.strip()
        if not p: continue
        if len(p) <= max_chars:
            frases.append(p); continue
        buf = ""
        for s in re.split(r"(?<=,)\s+", p):
            if buf and len(buf) + len(s) > max_chars:
                frases.append(buf.strip()); buf = s
            else:
                buf = (buf + " " + s).strip()
        if buf: frases.append(buf.strip())
    return frases


def _interpolar_tempos(palavras: list, total: float) -> list:
    """whisperx não marca tempo em alguns tokens (números, siglas): interpola entre vizinhos."""
    out = [[w, s, e] for w, s, e in palavras]; n = len(out)
    for i in range(n):
        if out[i][1] is not None: continue
        j0 = i
        while j0 > 0 and out[j0 - 1][1] is None: j0 -= 1
        prev_e = out[j0 - 1][2] if j0 > 0 else 0.0
        k = i
        while k < n and out[k][1] is None: k += 1
        next_s = out[k][1] if k < n else total
        passo = (next_s - prev_e) / max(k - j0, 1)
        for m in range(j0, k):
            out[m][1] = prev_e + passo * (m - j0); out[m][2] = prev_e + passo * (m - j0 + 1)
    return [(w, float(s), float(e)) for w, s, e in out]


class Sintetizador:
    """Interface única: sintetizar(texto, wav_destino) -> (duracao_s, [(palavra, inicio, fim), ...])."""

    def __init__(self, voz: str, amostra: pathlib.Path = None, cfg=CB_CFG, exag=CB_EXAG, permitir_cpu=False):
        self.voz, self.cfg, self.exag = voz, cfg, exag
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if voz not in VOZES_CLONADAS:
            return
        if self.device == "cpu" and not permitir_cpu:
            raise RuntimeError("GPU não detectada (torch.cuda.is_available() == False); na CPU cada slide leva ~10 min. "
                               "Reinstale o torch com CUDA ou use --permitir-cpu.")
        self.amostra = pathlib.Path(amostra or AMOSTRA_PADRAO)
        if not self.amostra.exists():
            raise FileNotFoundError(f"Amostra de voz não encontrada: {self.amostra}")
        self.ref = self._referencia_24k(self.amostra)
        if voz == "reinaldo_haas":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            print(f"Carregando Chatterbox Multilingual ({self.device})...")
            self.tts = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            self.sr = self.tts.sr
            self.tts.prepare_conditionals(str(self.ref), exaggeration=exag)   # codifica a amostra uma vez
        else:
            self.ref, self.ref_text = self._referencia_f5(self.ref)
            self._carregar_f5()
        import whisperx
        self.align_dev = "cuda" if (self.device == "cuda" and
                                    torch.cuda.get_device_properties(0).total_memory > 12e9) else "cpu"
        print(f"Carregando alinhador whisperx (pt) na {self.align_dev.upper()}...")
        self.align_model, self.align_meta = whisperx.load_align_model(language_code="pt", device=self.align_dev)
        print(f"Voz clonada ativa ({voz}) a partir de {self.amostra.name}")

    # ---- referência ----
    @staticmethod
    def _referencia_24k(amostra: pathlib.Path) -> pathlib.Path:
        wav = amostra.with_name(amostra.stem + "_ref24k.wav")
        if not wav.exists() or wav.stat().st_mtime < amostra.stat().st_mtime:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(amostra), "-ar", "24000", "-ac", "1",
                            "-af", "loudnorm=I=-20:TP=-2", str(wav)], check=True)
        loc = tmp_local(wav.name); shutil.copyfile(wav, loc)
        return loc

    @staticmethod
    def _referencia_f5(ref: pathlib.Path):
        """Trecho curto (<= F5_REF_S s, sem silêncio inicial) + transcrição em .txt ao lado da amostra.
        EDITE o .txt se o Whisper errar: o F5 depende dele."""
        clip = AMOSTRA_PADRAO.parent / f"f5_ref{F5_REF_S}s.wav"
        txt = clip.with_suffix(".txt")
        if not clip.exists() or clip.stat().st_mtime < ref.stat().st_mtime:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(ref), "-af",
                            "silenceremove=start_periods=1:start_threshold=-40dB:start_silence=0.2",
                            "-t", str(F5_REF_S), "-ar", "24000", "-ac", "1", str(clip)], check=True)
            if txt.exists(): txt.unlink()
        if not (txt.exists() and txt.read_text(encoding="utf-8").strip()):
            print("Transcrevendo a referência uma vez (faster-whisper)...")
            from faster_whisper import WhisperModel
            segs, _ = WhisperModel("small", device="cpu", compute_type="int8").transcribe(str(clip), language="pt")
            txt.write_text(" ".join(s.text.strip() for s in segs), encoding="utf-8")
            print(f"  salvo em {txt} — confira o texto.")
        loc = tmp_local(clip.name); shutil.copyfile(clip, loc)
        return loc, txt.read_text(encoding="utf-8").strip()

    def _carregar_f5(self):
        from huggingface_hub import snapshot_download
        from f5_tts.api import F5TTS
        pasta = pathlib.Path(snapshot_download(F5_REPO, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1"))
        ckpt = max(list(pasta.rglob("*.safetensors")) + list(pasta.rglob("*.pt")), key=lambda f: f.stat().st_size)
        vocab = next(iter(pasta.rglob("vocab.txt")), None)
        print(f"Carregando F5-TTS pt-BR ({self.device}): {ckpt.name}")
        erro = None
        for arq in ("F5TTS_Base", "F5TTS_v1_Base"):
            try:
                self.tts = F5TTS(model=arq, ckpt_file=str(ckpt), vocab_file=str(vocab or ""), device=self.device)
                return
            except Exception as e:
                erro = e
        raise RuntimeError(f"F5-TTS não carregou com nenhuma arquitetura conhecida: {erro}")

    # ---- geração ----
    def _frase(self, frase: str) -> torch.Tensor:
        with torch.inference_mode():
            if self.voz == "reinaldo_haas":
                return self.tts.generate(frase, language_id="pt", exaggeration=self.exag,
                                         cfg_weight=self.cfg, temperature=CB_TEMP).cpu()
            import numpy as np
            wav, sr, _ = self.tts.infer(ref_file=str(self.ref), ref_text=self.ref_text, gen_text=frase,
                                        nfe_step=F5_NFE, cfg_strength=F5_CFG, speed=F5_SPEED,
                                        remove_silence=False, show_info=lambda *a, **k: None, progress=None)
            self.sr = sr
            t = torch.from_numpy(np.asarray(wav, dtype=np.float32))
            return t.unsqueeze(0) if t.dim() == 1 else t

    def _clonado(self, texto: str, wav_out: pathlib.Path):
        import torchaudio
        pedacos = []
        for f in dividir_em_frases(texto):
            pedacos += [self._frase(f), torch.zeros(1, int(PAUSA_FRASES * self.sr))]
        audio = torch.cat(pedacos[:-1], dim=1)
        audio = audio / audio.abs().max().clamp(min=1e-6) * 0.9
        torchaudio.save(str(wav_out), audio, self.sr)
        if self.device == "cuda": torch.cuda.empty_cache()
        return audio.shape[1] / self.sr

    def alinhar(self, texto: str, wav: pathlib.Path, total: float) -> list:
        """Alinhamento forçado: o texto é conhecido, só recuperamos os tempos das palavras."""
        import whisperx
        audio = whisperx.load_audio(str(wav))
        res = whisperx.align([{"text": texto, "start": 0.0, "end": total}], self.align_model, self.align_meta,
                             audio, self.align_dev, return_char_alignments=False)
        pal = [(w.get("word", "").strip(), w.get("start"), w.get("end")) for w in res.get("word_segments", [])]
        pal = [p for p in pal if p[0]]
        if not pal:
            toks = texto.split(); passo = total / max(len(toks), 1)
            return [(t, i * passo, (i + 1) * passo) for i, t in enumerate(toks)]
        return _interpolar_tempos(pal, total)

    async def _edge(self, texto: str, wav_out: pathlib.Path):
        import edge_tts
        voz = "pt-BR-FranciscaNeural" if self.voz == "francisca" else "pt-BR-AntonioNeural"
        c = edge_tts.Communicate(texto, voz, boundary="WordBoundary")
        mp3 = tmp_local("edge.mp3"); dados = bytearray(); palavras = []
        async for ch in c.stream():
            if ch["type"] == "audio": dados.extend(ch["data"])
            elif ch["type"] == "WordBoundary":
                s = ch["offset"] / 1e7; palavras.append((ch["text"], s, s + ch["duration"] / 1e7))
        mp3.write_bytes(dados)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ar", "24000", "-ac", "1", str(wav_out)], check=True)
        return palavras

    async def sintetizar(self, texto: str, wav_out: pathlib.Path):
        loc = tmp_local(wav_out.name)
        if self.voz in VOZES_CLONADAS:
            t0 = time.time(); dur = self._clonado(texto, loc); t1 = time.time()
            palavras = self.alinhar(texto, loc, dur)
            print(f"  [tempos] síntese {t1-t0:.1f}s | alinhamento {time.time()-t1:.1f}s | áudio {dur:.1f}s")
        else:
            palavras = await self._edge(texto, loc); dur = duracao(loc)
        shutil.copyfile(loc, wav_out); loc.unlink(missing_ok=True)
        return dur, palavras

    def realinhar(self, texto: str, wav: pathlib.Path):
        """Áudio em cache sem tempos: recupera só o alinhamento."""
        loc = tmp_local(wav.name); shutil.copyfile(wav, loc)
        dur = duracao(loc)
        pal = self.alinhar(texto, loc, dur) if self.voz in VOZES_CLONADAS else asyncio.run(self._edge(texto, loc))
        loc.unlink(missing_ok=True)
        return dur, pal

    def assinatura(self) -> dict:
        if self.voz == "reinaldo_haas":
            return {"motor": "chatterbox", "amostra": self.amostra.name, "mtime": int(self.amostra.stat().st_mtime),
                    "cfg": self.cfg, "exag": self.exag, "temp": CB_TEMP}
        if self.voz == "reinaldo_f5":
            return {"motor": "f5-ptbr", "amostra": self.amostra.name, "mtime": int(self.amostra.stat().st_mtime),
                    "ref_text": self.ref_text, "nfe": F5_NFE, "cfg": F5_CFG, "speed": F5_SPEED}
        return {"motor": f"edge:{self.voz}"}


# ================================================================== 3. LEGENDAR
def format_ass_time(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    cs = min(99, int(round((t - int(t)) * 100)))
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



# ================================================================== 4. RENDERIZAR
def renderizar(img: pathlib.Path, wav: pathlib.Path, ass: pathlib.Path, mp4: pathlib.Path) -> subprocess.Popen:
    """Dispara o ffmpeg em segundo plano (codificação na CPU sobreposta à síntese do próximo slide).
    Entradas copiadas para disco local: o '-loop 1' relê o PNG a cada quadro."""
    loc = {p: tmp_local(p.name) for p in (img, wav, ass)}
    for p, l in loc.items(): shutil.copyfile(p, l)
    ass_f = str(loc[ass]).replace("\\", "/").replace(":", "\\:")
    saida = tmp_local(mp4.name)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", str(FPS_CLIPE), "-i", str(loc[img]),
           "-i", str(loc[wav]), "-vf", f"subtitles='{ass_f}'", "-r", str(FPS_CLIPE),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-tune", "stillimage", "-threads", str(FFMPEG_THREADS),
           "-c:a", "aac", "-b:a", "160k", "-pix_fmt", "yuv420p", "-shortest", str(saida)]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    p._videoaula = (saida, mp4, list(loc.values()), time.time())
    return p


def concluir_render(p: subprocess.Popen):
    saida, mp4, temporarios, t0 = p._videoaula
    _, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg falhou em {mp4.name}: {err.decode(errors='replace')[-300:]}")
    shutil.move(str(saida), str(mp4))
    for t in temporarios: pathlib.Path(t).unlink(missing_ok=True)
    print(f"  clipe {mp4.name} pronto ({time.time()-t0:.0f}s)")


def concatenar(clipes: list, saida: pathlib.Path):
    lista = saida.with_suffix(".txt")
    lista.write_text("".join(f"file '{str(c.resolve()).replace(chr(92), '/')}'\n" for c in clipes), encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lista),
                    "-c", "copy", str(saida)], check=True)
    lista.unlink()
    print(f"\nVÍDEO GERADO: {saida}  ({saida.stat().st_size/1e6:.1f} MB)")


# ================================================================== PIPELINE
def carregar_roteiro(pptx, work, roteiro_path, slides):
    img_dir = work / "slides_img"
    if roteiro_path:
        roteiro = json.loads(pathlib.Path(roteiro_path).read_text(encoding="utf-8-sig"))
        if slides: roteiro = [r for r in roteiro if r["slide"] in slides]
        faltam = [r["slide"] for r in roteiro if _precisa_png(img_dir / f"slide_{r['slide']:03d}.png", pptx)]
        if faltam: extrair(pptx, img_dir, faltam)
    else:
        roteiro = extrair(pptx, img_dir, slides)
        (work / "roteiro_gerado_automaticamente.json").write_text(
            json.dumps(roteiro, indent=2, ensure_ascii=False), encoding="utf-8")
    return [r for r in roteiro if r.get("texto_fala", "").strip()]


async def processar(pptx: pathlib.Path, roteiro: list, voz: Sintetizador, a, banner_ini: int, banner_fim: int) -> list:
    """Para cada slide: cache -> síntese -> legenda -> render. Devolve os clipes na ordem do roteiro."""
    work = pptx.parent / "video_work" / pptx.stem
    for d in ("audio", "legendas", "clipes"): (work / d).mkdir(parents=True, exist_ok=True)
    ass_atual = voz.assinatura()
    clipes, renders = [], []
    for r in roteiro:
        n, texto = r["slide"], r["texto_fala"].strip()
        wav, meta = work / "audio" / f"slide_{n:03d}.wav", work / "audio" / f"slide_{n:03d}.json"
        ass, img, mp4 = work / "legendas" / f"slide_{n:03d}.ass", work / "slides_img" / f"slide_{n:03d}.png", work / "clipes" / f"clip_{n:03d}.mp4"
        print(f"\n[Slide {n:03d}] {texto[:80]}...")

        # ---- cache: mesmo texto, mesmo motor/amostra/parâmetros ----
        dados = None
        if not a.forcar and wav.exists() and meta.exists():
            try:
                d = json.loads(meta.read_text(encoding="utf-8"))
                if d.get("texto_fala") == texto and d.get("assinatura") == ass_atual: dados = d
            except Exception: pass
        if dados:
            dur, palavras = dados["dur"], dados["words_timing"]
            print(f"  áudio em cache ({dur:.1f}s)")
        else:
            dur, palavras = await voz.sintetizar(texto, wav)
            meta.write_text(json.dumps({"dur": dur, "texto_fala": texto, "words_timing": palavras,
                                        "assinatura": ass_atual}, ensure_ascii=False), encoding="utf-8")

        banner = a.aviso_inicio if n == banner_ini else a.aviso_fim if n == banner_fim else None
        ass.write_text(gerar_ass_wordboundary(palavras, r.get("texto_original", texto), 7, banner,
                                              a.duracao_aviso, dur, a.offset), encoding="utf-8")
        renders.append(renderizar(img, wav, ass, mp4)); clipes.append(mp4)
        while sum(p.poll() is None for p in renders) > 2: time.sleep(0.5)
        for p in [p for p in renders if p.poll() is not None]: concluir_render(p); renders.remove(p)
    for p in renders: concluir_render(p)
    return clipes


def saida_padrao(pptx: pathlib.Path, roteiro: list, filtrado: bool) -> pathlib.Path:
    d = pptx.parent / "video"; d.mkdir(exist_ok=True)
    a, b = roteiro[0]["slide"], roteiro[-1]["slide"]
    return d / (f"{pptx.stem}.mp4" if not filtrado else f"{pptx.stem}_slides_{a}_a_{b}.mp4")


# ================================================================== GPUs e PARALELO
def quadro_gpus(leituras=2, intervalo=3.0) -> dict:
    """{indice: (gb_livres, util_%)} — mínimo de livres e máximo de uso ao longo das leituras."""
    def ler():
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return {int(i): (float(f) / 1024, int(u)) for i, f, u in (l.split(",") for l in out.strip().splitlines())}
    ls = [ler()]
    for _ in range(leituras - 1): time.sleep(intervalo); ls.append(ler())
    return {i: (min(l[i][0] for l in ls), max(l[i][1] for l in ls)) for i in ls[0]}


def escolher_gpu(pedido: str):
    """Define CUDA_VISIBLE_DEVICES: 'auto' = placa mais ociosa com >= 8 GB livres; índice fixa; 'todas' não mexe."""
    if pedido == "todas": return
    if pedido != "auto": os.environ["CUDA_VISIBLE_DEVICES"] = pedido; return
    if os.environ.get("CUDA_VISIBLE_DEVICES"): return
    try: q = quadro_gpus()
    except Exception: return
    if len(q) <= 1: return
    print("GPUs: " + "  ".join(f"[{i}] {g[0]:.0f}GB livres/{g[1]}%" for i, g in sorted(q.items())))
    cand = [(u, -l, i) for i, (l, u) in q.items() if l >= 8] or [(u, -l, i) for i, (l, u) in q.items()]
    i = sorted(cand)[0][2]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i); print(f"GPU escolhida: {i}")


def paralelo(a, pptx: pathlib.Path, roteiro: list, roteiro_path: pathlib.Path, slides: list):
    """N trabalhadores espalhados pelas GPUs ociosas (máx. --por-gpu por placa: sem MPS, processos na
    MESMA placa se revezam e um modelo autorregressivo ganha pouco). O pai só concatena."""
    work = pptx.parent / "video_work" / pptx.stem
    if not WINDOWS:
        outros = [p for p in subprocess.run(["pgrep", "-f", f"gerador_de_video_e_som.py.*{re.escape(pptx.name)}.*--so-clipes"],
                                            capture_output=True, text=True).stdout.split() if int(p) != os.getpid()]
        if outros:
            print(f"ERRO: já há {len(outros)} trabalhador(es) desta aula rodando (PIDs {' '.join(outros)}). "
                  "Acompanhe ou encerre com: pkill -f gerador_de_video_e_som.py"); return
    # plano de GPUs
    VRAM_GB = 6.0
    try: q = quadro_gpus()
    except Exception: q = {}
    slots = []
    if a.gpu not in ("auto", "todas"):
        for i in a.gpu.split(","): slots += [int(i)] * a.por_gpu
    elif q:
        print("GPUs: " + "  ".join(f"[{i}] {g[0]:.0f}GB livres/{g[1]}%" for i, g in sorted(q.items())))
        for u, _, i in sorted((u, -l, i) for i, (l, u) in q.items() if u <= a.gpu_util_max and l >= VRAM_GB):
            slots += [i] * min(a.por_gpu, int(q[i][0] / VRAM_GB))
    if not slots: slots = [None]
    por = {}
    for s in slots: por.setdefault(s, []).append(s)
    slots = []
    while any(por.values()):
        for g in list(por):
            if por[g]: slots.append(por[g].pop())
    n = min(a.paralelo, len(slots), len(roteiro)); slots = slots[:n]
    # fatias balanceadas pelo tamanho do texto
    fatias, carga = [[] for _ in range(n)], [0] * n
    for r in sorted(roteiro, key=lambda r: -len(r["texto_fala"])):
        k = carga.index(min(carga)); fatias[k].append(r["slide"]); carga[k] += len(r["texto_fala"])
    primeiro, ultimo = roteiro[0]["slide"], roteiro[-1]["slide"]
    base = [sys.executable, str(pathlib.Path(__file__).resolve()), str(pptx), "--roteiro", str(roteiro_path),
            "--so-clipes", "--gpu", "todas", "--banner-primeiro", str(primeiro), "--banner-ultimo", str(ultimo),
            "-v", a.voz, "--cfg", str(a.cfg), "--exag", str(a.exag), "--offset", str(a.offset),
            "--duracao-aviso", str(a.duracao_aviso), "--aviso-inicio", a.aviso_inicio, "--aviso-fim", a.aviso_fim]
    if a.amostra: base += ["-a", a.amostra]
    if a.hf_home: base += ["--hf-home", a.hf_home]
    for f in ("forcar", "permitir_cpu", "online"):
        if getattr(a, f): base.append("--" + f.replace("_", "-"))
    logs = work / "logs"; logs.mkdir(parents=True, exist_ok=True)
    print(f"\nParalelo: {n} trabalhadores, {len(roteiro)} slides")
    procs, t0 = [], time.time()
    for k, (fatia, gpu) in enumerate(zip(fatias, slots), 1):
        env = os.environ.copy()
        if gpu is not None: env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = logs / f"trabalhador_{k:02d}.log"
        print(f"  [{k:02d}] GPU {gpu if gpu is not None else '-'}  {len(fatia)} slides  -> {log.name}")
        procs.append((k, subprocess.Popen(base + ["-s", ",".join(map(str, sorted(fatia)))],
                                          stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT, env=env)))
        time.sleep(3)
    clipes = [work / "clipes" / f"clip_{r['slide']:03d}.mp4" for r in roteiro]
    visto = -1
    while any(p.poll() is None for _, p in procs):
        time.sleep(10)
        prontos = sum(c.exists() for c in clipes)
        if prontos != visto: print(f"  {prontos}/{len(clipes)} clipes ({time.time()-t0:.0f}s)"); visto = prontos
    for k, p in procs:
        if p.returncode != 0:
            log = logs / f"trabalhador_{k:02d}.log"
            print(f"ERRO no trabalhador {k:02d} (veja {log}):\n   " +
                  "\n   ".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-6:])); return
    if any(not c.exists() for c in clipes):
        print("ERRO: faltam clipes:", [c.name for c in clipes if not c.exists()]); return
    print(f"Síntese concluída em {time.time()-t0:.0f}s.")
    concatenar(clipes, pathlib.Path(a.saida) if a.saida else saida_padrao(pptx, roteiro, bool(slides)))


# ================================================================== CLI
def parse_slides(s: str) -> list:
    out = set()
    for parte in (s or "").split(","):
        parte = parte.strip()
        if not parte: continue
        if "-" in parte:
            a, b = parte.split("-", 1); out.update(range(int(a or 1), int(b or 9999) + 1))
        else: out.add(int(parte))
    return sorted(out) or None


def main():
    p = argparse.ArgumentParser(description="Vídeo-aula a partir de .pptx com voz clonada (local).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("pptx", nargs="?", help="arquivo .pptx (padrão: o primeiro da pasta atual)")
    p.add_argument("-r", "--roteiro", help="roteiro JSON (opcional; senão vem das anotações do PPTX)")
    p.add_argument("-s", "--slides", help="filtro: '5', '3-7', '1,4,9', '10-'")
    p.add_argument("-v", "--voz", default="reinaldo_haas", choices=["reinaldo_haas", "reinaldo_f5", "antonio", "francisca"])
    p.add_argument("-a", "--amostra", help=f"áudio de referência da voz (padrão: {AMOSTRA_PADRAO.relative_to(AQUI)})")
    p.add_argument("--cfg", type=float, default=CB_CFG, help="Chatterbox: menor = mais fiel ao timbre")
    p.add_argument("--exag", type=float, default=CB_EXAG, help="Chatterbox: expressividade (0.5 neutro)")
    p.add_argument("-o", "--saida", help="arquivo .mp4 final")
    p.add_argument("--offset", type=float, default=-0.12, help="deslocamento das legendas (s)")
    p.add_argument("--aviso-inicio", default=DEFAULT_AVISO_INICIO); p.add_argument("--aviso-fim", default=DEFAULT_AVISO_FIM)
    p.add_argument("--duracao-aviso", type=float, default=5.0)
    p.add_argument("--sem-aviso-inicio", action="store_true"); p.add_argument("--sem-aviso-fim", action="store_true")
    p.add_argument("--forcar", action="store_true", help="ignora o cache de áudio")
    p.add_argument("--gpu", default="auto", help="'auto', índice (ex. 3) ou 'todas'")
    p.add_argument("-p", "--paralelo", type=int, default=1, metavar="N", help="trabalhadores em paralelo (várias GPUs)")
    p.add_argument("--por-gpu", type=int, default=2, help="máx. de trabalhadores por placa")
    p.add_argument("--gpu-util-max", type=int, default=15, help="só usa placas com uso <= este %%")
    p.add_argument("--permitir-cpu", action="store_true", help="aceita rodar sem GPU (muito lento)")
    p.add_argument("--online", action="store_true", help="permite baixar modelos do Hugging Face (1ª vez)")
    p.add_argument("--hf-home", help="pasta do cache de modelos (padrão ~/.cache/huggingface)")
    for h in ("--so-clipes",): p.add_argument(h, action="store_true", help=argparse.SUPPRESS)
    for h in ("--banner-primeiro", "--banner-ultimo"): p.add_argument(h, type=int, default=None, help=argparse.SUPPRESS)
    a = p.parse_args()

    # ambiente: modelos offline por padrão, GPU
    if a.hf_home: os.environ["HF_HOME"] = str(pathlib.Path(a.hf_home).resolve())
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if a.online: print("Modo ONLINE: modelos podem ser baixados do Hugging Face.")
    else: os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not a.so_clipes and a.paralelo <= 1: escolher_gpu(a.gpu)
    if a.sem_aviso_inicio: a.aviso_inicio = None
    if a.sem_aviso_fim: a.aviso_fim = None

    pptx = pathlib.Path(a.pptx) if a.pptx else next(iter(pathlib.Path(".").glob("*.pptx")), None)
    if not pptx or not pptx.exists(): sys.exit(f"Arquivo .pptx não encontrado: {a.pptx}")
    pptx = pptx.resolve()
    work = pptx.parent / "video_work" / pptx.stem; work.mkdir(parents=True, exist_ok=True)
    slides = parse_slides(a.slides)
    roteiro = carregar_roteiro(pptx, work, a.roteiro, slides)
    if not roteiro: sys.exit("Nenhum slide com texto para narrar.")
    roteiro_path = pathlib.Path(a.roteiro).resolve() if a.roteiro else work / "roteiro_gerado_automaticamente.json"

    if a.paralelo > 1 and not a.so_clipes:
        return paralelo(a, pptx, roteiro, roteiro_path, slides)

    gpu = torch.cuda.is_available()
    print(f"\n{len(roteiro)} slides ({roteiro[0]['slide']}..{roteiro[-1]['slide']}), voz '{a.voz}', "
          f"{'GPU ' + torch.cuda.get_device_name(0) if gpu else 'CPU'}, {N_CPU} CPUs")
    try:
        voz = Sintetizador(a.voz, a.amostra, a.cfg, a.exag, a.permitir_cpu)
    except Exception as e:
        if os.environ.get("HF_HUB_OFFLINE") == "1" and any(k in str(e) for k in ("offline", "cache", "LocalEntryNotFound", "not found")):
            sys.exit(f"Modelo ausente do cache local. Rode uma vez com --online.\n  ({str(e)[:200]})")
        raise
    banner_ini = a.banner_primeiro if a.banner_primeiro is not None else roteiro[0]["slide"]
    banner_fim = a.banner_ultimo if a.banner_ultimo is not None else roteiro[-1]["slide"]
    clipes = asyncio.run(processar(pptx, roteiro, voz, a, banner_ini, banner_fim))
    if a.so_clipes:
        print(f"[trabalhador] {len(clipes)} clipes prontos."); return
    concatenar(clipes, pathlib.Path(a.saida) if a.saida else saida_padrao(pptx, roteiro, bool(slides)))


if __name__ == "__main__":
    main()
