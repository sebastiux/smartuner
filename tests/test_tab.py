"""Pruebas de la etapa 8 (tablatura): emparejamiento, ASCII alineado y exportación."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from src.config import OPEN_STRING_MIDI
from src.environment import Arm
from src.segmentation import Segment
from src.tab import (
    CSV_COLUMNS,
    TAB_STRINGS,
    TIME_LABEL,
    TabNote,
    export_csv,
    export_json,
    export_tab,
    export_txt,
    notes_from_choices,
    render_ascii,
)

SR = 22050


def _note(i: int, start_s: float, string: str, fret: int, f0_hz: float | None = None) -> TabNote:
    """TabNote de prueba de 0.25 s en (cuerda, traste) con MIDI coherente."""
    return TabNote(position=i, start_s=start_s, end_s=start_s + 0.25, string=string, fret=fret,
                   midi=OPEN_STRING_MIDI[string] + fret, f0_hz=f0_hz)


def _riff(n: int = 30) -> list[TabNote]:
    """Secuencia determinista que recorre las 4 cuerdas con trastes de 1 y 2 dígitos."""
    strings = ("E", "A", "D", "G", "A", "E", "D")
    frets = (0, 3, 5, 12, 7, 10, 2, 0, 9)
    return [_note(i, 0.5 + 0.3 * i, strings[i % len(strings)], frets[i % len(frets)]) for i in range(n)]


def _segment(index: int, start_s: float, end_s: float, f0_hz: float | None) -> Segment:
    """Segmento conservado de prueba con f0 opcional."""
    return Segment(index=index, start_s=start_s, end_s=end_s, start_sample=int(start_s * SR),
                   end_sample=int(end_s * SR), rms_db=-6.0, f0_hz=f0_hz)


def _systems(text: str) -> list[list[str]]:
    """Bloques separados por líneas en blanco que contienen las 4 cuerdas (ignora el título)."""
    blocks = [b.split("\n") for b in text.split("\n\n")]
    return [b for b in blocks if any(line.lstrip().startswith("G|") for line in b)]


def _string_rows(system: list[str]) -> dict[str, str]:
    """Filas de cuerdas de un sistema de tablatura, por letra de cuerda."""
    rows = {}
    for line in system:
        m = re.match(r"^\s*([GDAE])\|", line)
        if m:
            rows[m.group(1)] = line
    return rows


def _parse_system(system: list[str]) -> list[tuple[str, int, int]]:
    """Lee las columnas de un sistema: lista de (cuerda, traste, columna_del_primer_dígito)."""
    rows = _string_rows(system)
    start = rows["G"].index("|") + 1
    end = len(rows["G"]) - 1
    out: list[tuple[str, int, int]] = []
    pos = start
    while pos < end:
        hits = [(s, r) for s, r in rows.items() if r[pos + 1].isdigit()]
        assert len(hits) == 1, f"columna {pos}: una nota debe estar en exactamente una cuerda"
        string, row = hits[0]
        digits = re.match(r"\d+", row[pos + 1:]).group(0)
        out.append((string, int(digits), pos + 1))
        width = len(digits) + 2
        for s, r in rows.items():
            cell = r[pos:pos + width]
            assert cell == (f"-{digits}-" if s == string else "-" * width)
        pos += width
    return out


# ---------------------------------------------------------------------------
# Emparejamiento segmentos ↔ brazos
# ---------------------------------------------------------------------------


def test_notes_from_choices_pairs_segments_and_arms() -> None:
    """notes_from_choices empareja segmentos y posiciones: posición entre conservados, tiempos, MIDI y f0."""
    segments = [_segment(2, 0.5, 0.8, 55.3), _segment(5, 0.8, 1.1, None)]
    arms = [Arm("A", 0), Arm("E", 5)]
    notes = notes_from_choices(segments, arms)
    assert [n.position for n in notes] == [0, 1]  # índice entre los conservados, no Segment.index
    assert [(n.string, n.fret, n.midi) for n in notes] == [("A", 0, 33), ("E", 5, 33)]
    assert notes[0].start_s == 0.5 and notes[0].end_s == 0.8
    assert notes[0].f0_hz == 55.3 and notes[1].f0_hz is None
    assert notes[1].label == "E-5" and notes[1].note_name == "A1"


def test_notes_from_choices_requires_same_length() -> None:
    """Distinto número de segmentos y posiciones es un error."""
    with pytest.raises(ValueError, match="segmentos"):
        notes_from_choices([_segment(0, 0.5, 0.8, 55.0)], [])


# ---------------------------------------------------------------------------
# Tablatura ASCII
# ---------------------------------------------------------------------------


def test_string_order_and_a0_on_a_line() -> None:
    """La tablatura imprime G, D, A, E (aguda arriba) y cada traste en su cuerda, en columnas sucesivas."""
    notes = [_note(0, 0.5, "A", 0), _note(1, 0.8, "D", 2), _note(2, 1.1, "G", 4), _note(3, 1.4, "E", 3)]
    system = _systems(render_ascii(notes))[0]
    labels = [re.match(r"^\s*([GDAE])\|", line).group(1) for line in system[1:]]
    assert labels == ["G", "D", "A", "E"] == list(TAB_STRINGS)
    rows = _string_rows(system)
    col = rows["A"].index("0")
    assert rows["A"][col - 1:col + 2] == "-0-"
    assert all(rows[s][col] == "-" for s in "GDE")
    assert _parse_system(system) == [("A", 0, col), ("D", 2, col + 3), ("G", 4, col + 6), ("E", 3, col + 9)]


@pytest.mark.parametrize("show_times", [True, False])
@pytest.mark.parametrize("line_width", [30, 50, 80])
def test_systems_are_aligned_and_split_at_line_width(show_times: bool, line_width: int) -> None:
    """Los sistemas no superan el ancho de línea, todas sus líneas miden lo mismo y ninguna nota se pierde al partir."""
    notes = _riff(30)
    text = render_ascii(notes, line_width=line_width, show_times=show_times, title="Prueba")
    systems = _systems(text)
    assert len(systems) > 1
    parsed: list[tuple[str, int]] = []
    for system in systems:
        assert len(system) == (5 if show_times else 4)
        lengths = {len(line) for line in system}
        assert len(lengths) == 1, f"líneas de distinto largo: {system}"
        assert lengths.pop() <= line_width
        for line in _string_rows(system).values():
            assert line.endswith("|")
        parsed += [(s, f) for s, f, _ in _parse_system(system)]
    # Ninguna nota se pierde ni se duplica al partir en sistemas.
    assert parsed == [(n.string, n.fret) for n in notes]


def test_time_row_aligns_first_and_last_note_of_each_system() -> None:
    """La fila de tiempos alinea el tiempo de la primera y la última nota de cada sistema con su traste."""
    notes = _riff(30)
    systems = _systems(render_ascii(notes, line_width=50))
    k = 0
    for system in systems:
        cols = _parse_system(system)
        time_row = system[0]
        assert time_row.startswith(TIME_LABEL)
        first, last = notes[k], notes[k + len(cols) - 1]
        first_txt, last_txt = f"{first.start_s:.2f}", f"{last.start_s:.2f}"
        # El tiempo de la primera nota empieza justo sobre su traste.
        assert time_row[cols[0][2]:cols[0][2] + len(first_txt)] == first_txt
        # El de la última aparece y se solapa con su columna.
        idx = time_row.rfind(last_txt)
        assert idx >= 0 and idx <= cols[-1][2] + len(str(last.fret)) and idx + len(last_txt) > cols[-1][2] - 1
        # Todas las marcas son tiempos de notas del sistema, separadas por espacios.
        tokens = time_row[len(TIME_LABEL):].split()
        times = {f"{n.start_s:.2f}" for n in notes[k:k + len(cols)]}
        assert set(tokens) <= times and len(tokens) >= 2
        k += len(cols)
    assert k == len(notes)


def test_time_labels_are_placed_over_their_fret_when_space_allows() -> None:
    # Trastes de 2 dígitos (columnas de 4) con tiempos de 4 caracteres: caben casi todos.
    """Cuando hay sitio, cada tiempo se escribe justo encima de su traste."""
    notes = [_note(i, 1.0 + i, "A", 10 + i % 3) for i in range(6)]
    system = _systems(render_ascii(notes, line_width=80))[0]
    cols = _parse_system(system)
    time_row = system[0]
    shown = 0
    for (_, _, col), note in zip(cols, notes):
        txt = f"{note.start_s:.2f}"
        if time_row[col:col + len(txt)] == txt:
            shown += 1
    assert shown >= 3


def test_title_two_digit_frets_and_no_times() -> None:
    """Título en la primera línea, trastes de dos dígitos y opción sin fila de tiempos."""
    notes = [_note(0, 0.5, "G", 12), _note(1, 0.9, "E", 0)]
    text = render_ascii(notes, show_times=False, title="Mi bajo")
    lines = text.split("\n")
    assert lines[0] == "Mi bajo"
    assert TIME_LABEL not in text
    system = _systems(text)[0]
    rows = _string_rows(system)
    assert rows["G"] == "G|-12----|"
    assert rows["D"] == "D|-------|"
    assert rows["E"] == "E|-----0-|"


def test_tiny_line_width_puts_one_note_per_system() -> None:
    """Con un ancho mínimo cada sistema lleva una sola nota (nunca se pierde ninguna)."""
    notes = _riff(5)
    systems = _systems(render_ascii(notes, line_width=5))
    assert len(systems) == 5
    assert [len(_parse_system(s)) for s in systems] == [1] * 5


def test_empty_tab_and_invalid_string() -> None:
    """Una tablatura vacía se dibuja alineada y una cuerda desconocida es un error."""
    text = render_ascii([])
    system = _systems(text)[0]
    assert len({len(line) for line in system}) == 1
    with pytest.raises(ValueError):
        render_ascii([TabNote(0, 0.0, 0.1, "B", 0, 23)])


# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------


def test_export_txt_json_csv_are_readable(tmp_path: Path) -> None:
    """Las exportaciones TXT, JSON y CSV se releen con los mismos datos (tiempos, MIDI, nota, f0)."""
    notes = [_note(0, 0.5, "A", 0, f0_hz=55.12), _note(1, 0.8, "D", 2), _note(2, 1.1, "E", 5, f0_hz=55.0)]

    txt = export_txt(notes, tmp_path / "out" / "tab.txt", title="Título")
    assert txt.read_text(encoding="utf-8") == render_ascii(notes, title="Título") + "\n"

    js = export_json(notes, tmp_path / "tab.json", meta={"algoritmo": "ucb1", "λ": 0.1})
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["meta"] == {"algoritmo": "ucb1", "λ": 0.1}
    assert [(d["string"], d["fret"], d["midi"], d["note"]) for d in data["notes"]] == [
        ("A", 0, 33, "A1"), ("D", 2, 40, "E2"), ("E", 5, 33, "A1")]
    assert data["notes"][1]["f0_hz"] is None and data["notes"][0]["f0_hz"] == pytest.approx(55.12)
    restored = [TabNote(**{k: d[k] for k in ("position", "start_s", "end_s", "string", "fret", "midi", "f0_hz")})
                for d in data["notes"]]
    assert [(n.position, n.label, n.midi) for n in restored] == [(n.position, n.label, n.midi) for n in notes]
    assert [n.start_s for n in restored] == pytest.approx([n.start_s for n in notes])
    assert [n.end_s for n in restored] == pytest.approx([n.end_s for n in notes])

    cs = export_csv(notes, tmp_path / "tab.csv")
    with cs.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
        rows = list(reader)
    assert [r["note"] for r in rows] == ["A1", "E2", "A1"]
    assert [int(r["fret"]) for r in rows] == [0, 2, 5]
    assert rows[1]["f0_hz"] == "" and float(rows[0]["f0_hz"]) == pytest.approx(55.12)
    assert float(rows[2]["start_s"]) == pytest.approx(1.1)


def test_export_tab_dispatches_by_extension(tmp_path: Path) -> None:
    """export_tab elige el formato por la extensión (TXT/JSON/CSV) y rechaza las desconocidas."""
    notes = _riff(4)
    meta = {"title": "Riff de prueba", "algoritmo": "softmax"}
    txt = export_tab(notes, tmp_path / "a.TXT", meta=meta)
    assert txt.read_text(encoding="utf-8").startswith("Riff de prueba\n")
    assert json.loads(export_tab(notes, tmp_path / "a.json", meta=meta).read_text(encoding="utf-8"))["meta"] == meta
    assert export_tab(notes, tmp_path / "a.csv").read_text(encoding="utf-8").startswith(",".join(CSV_COLUMNS))
    # Sin "title" el TXT usa un resumen de los metadatos.
    assert "algoritmo: softmax" in export_tab(notes, tmp_path / "b.txt", meta={"algoritmo": "softmax"}).read_text(
        encoding="utf-8").split("\n")[0]
    with pytest.raises(ValueError):
        export_tab(notes, tmp_path / "a.pdf")
