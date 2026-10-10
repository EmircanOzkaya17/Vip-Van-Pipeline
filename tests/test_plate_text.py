import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.plate_text import correct_and_validate_plate as v, majority_vote_plate as mv

cases = {
    # eski davranis korunmali
    "34 ABC 123": "34ABC123", "06AB1234": "06AB1234", "35 A 12345": "35A12345",
    "34A8C123": "34ABC123",   # 8 -> B duzeltmesi (harf bolgesi)
    "O6AB1234": "06AB1234",   # O -> 0 (il kodu)
    # yeni: TR ulke kodu / fazladan karakter
    "TR34ABC123": "34ABC123", "TR 06 AB 1234": "06AB1234", "TR35A12345": "35A12345",
    "34ABC123|": "34ABC123", "IR34ABC123": "34ABC123", "34ABC1231": "34ABC123",
    # reddedilmeli
    "": None, "ABCDEFGH": None, "TR": None, "1234": None,
    "MERCEDES BENZ VITO TOURER": None, "99ABC123": None,
}
bad = 0
for raw, want in cases.items():
    got = v(raw)
    ok = got == want
    bad += not ok
    print(f"{'OK ' if ok else 'XX '} {raw!r:30} -> {got!r:12} (beklenen {want!r})")

print()
vote_cases = [
    ([("34ABC123", .6), ("34ABC123", .5)], "34ABC123"),
    ([("34ABC123", .9)], "34ABC123"),
    ([("34ABC123", .6)], None),
    ([("34ABC123", .6), ("34ABC128", .6), ("34ABG123", .5)], "34ABC123"),   # karakter oylamasi
    ([("34ABC123", .6), ("06AB1234", .6), ("35A12345", .5)], None),        # uzlasma yok
]
for cands, want in vote_cases:
    got, conf = mv(cands)
    ok = got == want
    bad += not ok
    print(f"{'OK ' if ok else 'XX '} {[c for c,_ in cands]} -> {got} ({conf})")
print("\nHATA:", bad)
sys.exit(1 if bad else 0)
