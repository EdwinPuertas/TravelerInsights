"""Simula el paquete que sube Vercel (respetando .vercelignore) y verifica que la API arranque."""
import fnmatch
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ignorado(rel: str, patrones: list[str]) -> bool:
    for p in patrones:
        anclado = p.startswith("/")
        p = p.lstrip("/")
        if p.endswith("/"):
            d = p.rstrip("/")
            partes = rel.split("/")[:-1]
            if anclado and partes[: len(d.split("/"))] == d.split("/"):
                return True
            if not anclado and d in partes:
                return True
        elif (anclado and fnmatch.fnmatch(rel, p)) or (not anclado and fnmatch.fnmatch(Path(rel).name, p)):
            return True
    return False


def test_bundle_vercel_arranca(tmp_path):
    patrones = [ln.strip() for ln in (ROOT / ".vercelignore").read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
    archivos = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    if not archivos:
        archivos = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    for rel in archivos:
        if not _ignorado(rel, patrones) and (ROOT / rel).exists():
            dst = tmp_path / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dst)
    assert (tmp_path / "src/traveler_insights/core/data/emociones.json").exists()
    codigo = ("import app; from fastapi.testclient import TestClient; c = TestClient(app.app); "
              "r = c.post('/api/analyze', json={'texts': ['buen servicio']}); "
              "assert c.get('/api/health').status_code == 200 and r.status_code in (200, 503)")
    res = subprocess.run([sys.executable, "-c", codigo], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
