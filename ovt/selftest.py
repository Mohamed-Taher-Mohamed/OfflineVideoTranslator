"""Self-tests the user runs on THEIR machine with THEIR installed models, to build the
real 'supported languages' register. Nothing here claims success unless the engine produced it."""
import re

SAMPLES = {
    "en": ["The meeting starts at nine o'clock tomorrow morning.", "I did not expect to see you here after all these years.", "Please close the door and turn off the lights."],
    "fr": ["La réunion commence demain matin à neuf heures.", "Je ne m'attendais pas à te voir ici après toutes ces années.", "Ferme la porte et éteins les lumières, s'il te plaît."],
    "es": ["La reunión empieza mañana por la mañana a las nueve.", "No esperaba verte aquí después de tantos años.", "Por favor, cierra la puerta y apaga las luces."],
    "de": ["Die Besprechung beginnt morgen früh um neun Uhr.", "Ich hätte nicht erwartet, dich nach all den Jahren hier zu sehen.", "Bitte schließe die Tür und mach das Licht aus."],
}
_AR = re.compile(r"[\u0600-\u06ff]")


def run_mt_selftest(translator, src: str, tgt: str = "ar") -> dict:
    sup = translator.supports(src, tgt)
    rep = {"src": src, "tgt": tgt, "supported": sup.ok, "route": sup.route, "reason": sup.reason, "samples": [], "passed": False}
    if not sup.ok or src not in SAMPLES:
        if sup.ok:
            rep["reason"] = "no built-in sample sentences for '%s'; test it manually" % src
        return rep
    out = translator.translate(SAMPLES[src], src, tgt)
    ok = len(out) == len(SAMPLES[src])
    for a, b in zip(SAMPLES[src], out):
        has_ar = bool(_AR.search(b)) if tgt == "ar" else bool(b.strip())
        rep["samples"].append({"source": a, "output": b, "has_target_script": has_ar})
        ok = ok and has_ar and b.strip() != a.strip()
    rep["passed"] = ok       # 'passed' = produced Arabic text. A human must still judge quality.
    return rep
