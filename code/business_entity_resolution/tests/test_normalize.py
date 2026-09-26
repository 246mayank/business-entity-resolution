import polars as pl
import pytest

from src import contracts as C
from src.normalize import normalize_frame, transliterate

TABLE = {"राम": "ram", "मार्केटिंग": "marketing", "प्राइवेट": "private", "लिमिटेड": "limited", "ಕರ್ನಾಟಕ": "karnataka"}


def norm(names, addrs=None):
    addrs = addrs or [""] * len(names)
    df = pl.DataFrame({"entity_id": [f"S2-{i}" for i in range(len(names))], "business_name": names,
                       "business_address": addrs, "country": ["x"] * len(names)})
    return normalize_frame(df, TABLE)


@pytest.mark.parametrize("raw, core, legal", [
    ("राम मार्केटिंग प्राइवेट लिमिटेड", "ram marketing", "ltd_pvt"),
    ("Pvt. EFS Print Ventures Ltd.", "efs print ventures", "ltd_pvt"),
    ("LLC Moncada Léarning Center", "moncada learning center", "llc"),
    ("-- Holloway Peak Inc Seafood", "holloway peak seafood", "inc"),
    ("Kolkata Metal [Prívate]", "kolkata metal", "pvt"),
    ("Infra Rajalakshmi Gallery [Corp]", "infra rajalakshmi gallery", "corp"),
    ("Smt Sunrise Projects (Private)", "sunrise projects", "pvt"),
    ("S.A.S Lyons & Cie", "lyons", "co_sas"),
    ("Corwin, Rhona J., O.D., M.D., P.C.", "corwin rhona j od md", "pc"),
    ("Precisi0n Rlarlty [Laboratories]", "precision rlarlty laboratories", ""),
    ("The Maki Partners (LLC)", "maki partners", "llc"),
    ("Arpege Amicale SARL", "arpege amicale", "sarl"),
])
def test_name_core_and_legal_form(raw, core, legal):
    out = norm([raw])
    assert out["name_core"][0] == core
    assert out["legal_form"][0] == legal


def test_name_that_is_only_a_legal_form_keeps_its_text():
    assert norm(["Limited"])["name_core"][0] == "limited"


def test_dba_keeps_real_name_and_alias():
    out = norm(["Lyraxylo Co DBA Harbor Esports L.L.C."])
    assert out["name_core"][0] == "harbor esports"
    assert out["name_alias"][0] == "lyraxylo co"


@pytest.mark.parametrize("raw, nospace", [
    ("M/s SURAJTRADINGPARTNERS.COM", "surajtradingpartners"),
    ("downtownautobody.com", "downtownautobody"),
    ("www.cure8all.com", "cure8all"),
])
def test_domains(raw, nospace):
    out = norm([raw])
    assert out["is_domain"][0]
    assert out["name_nospace"][0] == nospace


@pytest.mark.parametrize("raw, clean, house", [
    ("105 ELM ST, MORGANTON, NC", "105 elm street morganton nc", "105"),
    ("5780 Fawn Ct, Fort Worth, Texas", "5780 fawn court fort worth tx", "5780"),
    ("22727 MIRAMAR BEND DR, <NULL>, TOMBALL, TX", "22727 miramar bend drive tomball tx", "22727"),
    ("409C CHARLES AVENUE, NULL, SIDNEY, OH", "409c charles avenue sidney oh", "409c"),
    ("Chandi Ct, Mansfield CITY, Texas", "chandi court mansfield tx", ""),
    ("KH NO. -570/13, NEW DELHI, Delhi", "kh 570 13 new delhi delhi", "570/13"),
    ("Door No 183, Jayanagar, ಕರ್ನಾಟಕ", "door 183 jayanagar karnataka", "183"),
    ("99 R. Judaique, Bordeaux, Gironde", "99 rue judaique bordeaux gironde", "99"),
    ("N/A", "", ""),
    ("", "", ""),
])
def test_addresses(raw, clean, house):
    out = norm(["x"], [raw])
    assert out["addr_clean"][0] == clean
    assert out["house_no"][0] == house


def test_street_and_places():
    out = norm(["x", "y"], ["105 ELM ST, MORGANTON, NC", "MN, Isabella, 1298 Wander Road"])
    assert out["addr_street"].to_list() == ["105 elm street", "1298 wander road"]
    assert out["addr_places"].to_list() == [["morganton", "nc"], ["mn", "isabella"]]


@pytest.mark.parametrize("raw, house", [("0049 Rue Jean Mermoz, Calais", "49"), ("100 Main St", "100"), ("1005 A Rd", "1005")])
def test_leading_zeros(raw, house):
    assert norm(["x"], [raw])["house_no"][0] == house


def test_transliterate_uses_table_and_fallback():
    assert transliterate("Silver राम Agro", TABLE).split() == ["Silver", "ram", "Agro"]
    assert transliterate("‌राम‌", TABLE).strip() == "ram"
    assert transliterate("अज्ञात", TABLE).strip().isascii()


def test_output_is_contract_valid_and_ascii():
    names = ["राम मार्केटिंग प्राइवेट लिमिटेड", "Café Élysée SARL", "", "!!!"]
    out = norm(names, ["ಕರ್ನಾಟಕ", "3 Rue de l'Herminette, Nantes", "", "---"])
    C.check_c1(out)
    for col in ("name_clean", "name_core", "addr_clean"):
        assert out[col].str.contains(r"^[a-z0-9 ]*$").all(), col
    assert out["is_native_script"].to_list() == [True, False, False, False]
