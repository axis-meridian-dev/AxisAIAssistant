#!/usr/bin/env python3
"""
BULK LEGAL DOWNLOADER v2 — FIXED URLs
======================================
Fixes from v1:
  - CT statutes: title_53a.htm (underscore format)
  - CT chapters: chap_950.htm (direct chapter download)
  - Constitutional amendments: first_amendment, fourth_amendment (word format)
  - UN documents: alternate sources for blocked OHCHR pages
  - Statistics: updated BJS/FBI URLs

Usage:
  python bulk_download_v2.py --ct-statutes     # CT General Statutes (by chapter)
  python bulk_download_v2.py --amendments      # All Constitutional Amendments
  python bulk_download_v2.py --un              # UN/International (fixed sources)
  python bulk_download_v2.py --stats           # Statistics (fixed URLs)
  python bulk_download_v2.py --all             # Everything above
  python bulk_download_v2.py --ingest          # Ingest after download
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

BASE_DIR = Path.home() / "LegalResearch"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"}
RATE_LIMIT = 2.0

client = httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30, verify=False)

downloaded = 0
errors = 0

def info(msg): print(f"\033[0;32m[+]\033[0m {msg}", flush=True)
def warn(msg): print(f"\033[1;33m[!]\033[0m {msg}", flush=True)
def error(msg): print(f"\033[0;31m[✗]\033[0m {msg}", flush=True)

def save_text(path, content, url=""):
    global downloaded
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Source: {url}\n# Downloaded: {datetime.now().isoformat()}\n\n{content}")
    downloaded += 1
    print(f"    ✓ {path.name}", flush=True)

def fetch(url):
    global errors
    time.sleep(RATE_LIMIT)
    try:
        r = client.get(url)
        r.raise_for_status()
        return r.text
    except Exception as e:
        error(f"  {url} — {e}")
        errors += 1
        return None

def clean(html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script","style","nav","footer","header","aside"]):
        t.decompose()
    main = soup.find("main") or soup.find("article") or soup.find("div",id="content") or soup
    text = main.get_text(separator="\n", strip=True)
    return "\n".join(l.strip() for l in text.split("\n") if l.strip())


# ════════════════════════════════════════════════════════════════════════════
# CT GENERAL STATUTES — BY CHAPTER (correct URLs)
# ════════════════════════════════════════════════════════════════════════════

CT_CHAPTERS = {
    # Title 29 — Public Safety and State Police
    "chap_541": "State Police",
    "chap_529": "Division of State Police",
    # Title 46a — Human Rights  
    "chap_814c": "Human Rights and Opportunities",
    "chap_814e": "Discriminatory Practices",
    # Title 53 — Crimes
    "chap_939": "Offenses Against the Person",
    "chap_939a": "Hate Crimes",
    "chap_940": "Offenses Against Public Order",
    "chap_941": "Offenses Against Public Peace",
    "chap_942": "Offenses Against Morality",
    "chap_943": "Offenses Against Public Justice",
    "chap_944": "Offenses Against the Public Safety",
    "chap_945": "Offenses Against Humanity",
    "chap_946": "Offenses Against Public Policy",
    # Title 53a — Penal Code
    "chap_950": "Penal Code General Provisions and Definitions",
    "chap_951": "Statutory Construction and Criminal Liability",
    "chap_952": "Penal Code Offenses and Penalties",
    # Title 54 — Criminal Procedure
    "chap_959": "Court Jurisdiction and Power",
    "chap_960": "Arraignment and Bail",
    "chap_961": "Trial and Proceedings After Conviction",
    "chap_961a": "Criminal Records",
    "chap_962": "Probation and Conditional Discharge",
    "chap_963": "Parole",
    # Title 52 — Civil Actions
    "chap_925": "Summary Process (Eviction)",
    "chap_926": "Damages and Civil Remedies",
    # Title 46b — Family Law
    "chap_815t": "Family Relations and Support",
    # Title 47a — Landlord and Tenant
    "chap_830": "Rights and Responsibilities of Landlord and Tenant",
    # Title 14 — Motor Vehicles
    "chap_248": "Vehicle Highway Use",
    "chap_249": "Operation of Motor Vehicles",
    # Title 51 — Courts
    "chap_884": "Court Organization",
    "chap_885": "Superior Court Judges",
}


def download_ct_statutes():
    info("═══ CT GENERAL STATUTES (by chapter) ═══")
    save_dir = BASE_DIR / "state_statutes" / "connecticut"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Download each chapter directly
    for chap_id, chap_name in CT_CHAPTERS.items():
        url = f"https://www.cga.ct.gov/current/pub/{chap_id}.htm"
        info(f"  {chap_id}: {chap_name}")
        html = fetch(url)
        if html:
            text = clean(html)
            if len(text) > 100:
                safe = re.sub(r'[^\w\-]', '_', chap_name)[:50]
                save_text(save_dir / f"{chap_id}_{safe}.txt", text, url)
    
    # Also download title index pages
    for title in ["title_29", "title_46a", "title_51", "title_52", "title_53", "title_53a", "title_54", "title_14", "title_46b", "title_47a"]:
        url = f"https://www.cga.ct.gov/current/pub/{title}.htm"
        info(f"  {title} (index)")
        html = fetch(url)
        if html:
            text = clean(html)
            if len(text) > 100:
                save_text(save_dir / f"{title}_index.txt", text, url)
    
    # Key individual sections
    info("  Key individual sections...")
    sections = [
        ("sec_46a-58", "Deprivation of Rights - Hate Crimes"),
        ("sec_46a-64", "Discriminatory Housing Practices"),
        ("sec_46a-64c", "Discriminatory Public Accommodation"),
        ("sec_46a-71", "Discriminatory Employment Practices"),
        ("sec_53a-3", "Penal Code Definitions"),
        ("sec_53a-8", "Criminal Liability for Acts of Another"),
        ("sec_53a-19", "Use of Physical Force in Defense of Person"),
        ("sec_53a-22", "Use of Physical Force by Peace Officer"),
        ("sec_53a-54a", "Murder"),
        ("sec_53a-59", "Assault First Degree"),
        ("sec_53a-60", "Assault Second Degree"),
        ("sec_53a-61", "Assault Third Degree"),
        ("sec_53a-61a", "Assault of Elderly or Disabled"),
        ("sec_53a-73a", "Sexual Assault Fourth Degree"),
        ("sec_53a-167a", "Interfering with Officer"),
        ("sec_53a-167c", "Assault of Public Safety Officer"),
        ("sec_53a-217", "Criminal Possession of Firearm"),
        ("sec_54-1f", "Racial Profiling Prohibition"),
        ("sec_54-33a", "Search and Seizure"),
        ("sec_54-33b", "Search Warrants"),
        ("sec_54-33c", "Arrest Without Warrant"),
        ("sec_54-33f", "Unreasonable Searches Prohibited"),
        ("sec_54-1h", "Arrest by Complaint and Summons"),
        ("sec_54-63c", "Right to Bail"),
        ("sec_54-63d", "Release on Promise to Appear"),
        ("sec_54-76h", "Right to Counsel"),
        ("sec_54-82i", "Discovery Rights"),
        ("sec_29-6a", "State Police Use of Force Reporting"),
        ("sec_29-6d", "Police Body Cameras"),
        ("sec_7-282", "Municipal Police Training"),
        ("sec_7-277", "Appointment of Special Constables"),
        ("sec_7-294d", "Police Officer Standards and Training"),
        ("sec_52-571a", "Action for Civil Rights Violations"),
    ]
    
    for sec_id, sec_name in sections:
        url = f"https://www.cga.ct.gov/current/pub/{sec_id}.htm"
        html = fetch(url)
        if html:
            text = clean(html)
            if len(text) > 50:
                save_text(save_dir / f"{sec_id.replace('-','_')}_{sec_name.replace(' ','_')[:30]}.txt", text, url)


# ════════════════════════════════════════════════════════════════════════════
# CONSTITUTIONAL AMENDMENTS (correct Cornell URLs)
# ════════════════════════════════════════════════════════════════════════════

def download_amendments():
    info("═══ CONSTITUTIONAL AMENDMENTS ═══")
    save_dir = BASE_DIR / "federal_statutes"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Cornell uses word-based URLs: first_amendment, fourth_amendment, etc.
    amendments = [
        ("first_amendment", "1st Amendment - Free Speech Religion Assembly"),
        ("second_amendment", "2nd Amendment - Right to Bear Arms"),
        ("third_amendment", "3rd Amendment - Quartering of Troops"),
        ("fourth_amendment", "4th Amendment - Search and Seizure"),
        ("fifth_amendment", "5th Amendment - Due Process Self Incrimination"),
        ("sixth_amendment", "6th Amendment - Right to Counsel Speedy Trial"),
        ("seventh_amendment", "7th Amendment - Jury Trial Civil Cases"),
        ("eighth_amendment", "8th Amendment - Cruel and Unusual Punishment"),
        ("ninth_amendment", "9th Amendment - Non Enumerated Rights"),
        ("tenth_amendment", "10th Amendment - Rights Reserved to States"),
        ("amendmentxi", "11th Amendment - Sovereign Immunity"),
        ("amendmentxii", "12th Amendment - Electoral College"),
        ("amendmentxiii", "13th Amendment - Abolition of Slavery"),
        ("amendmentxiv", "14th Amendment - Equal Protection Due Process"),
        ("amendmentxv", "15th Amendment - Voting Rights Race"),
        ("amendmentxvi", "16th Amendment - Income Tax"),
        ("amendmentxvii", "17th Amendment - Direct Election of Senators"),
        ("amendmentxix", "19th Amendment - Womens Suffrage"),
        ("amendmentxxiv", "24th Amendment - Poll Tax Abolished"),
        ("amendmentxxvi", "26th Amendment - Voting Age 18"),
    ]
    
    for url_slug, name in amendments:
        url = f"https://www.law.cornell.edu/constitution/{url_slug}"
        info(f"  {name}")
        html = fetch(url)
        if html:
            text = clean(html)
            if len(text) > 50:
                safe = re.sub(r'[^\w\-]', '_', name)[:50]
                save_text(save_dir / f"amendment_{safe}.txt", text, url)
    
    # Bill of Rights overview
    url = "https://www.law.cornell.edu/constitution/billofrights"
    info("  Bill of Rights (overview)")
    html = fetch(url)
    if html:
        save_text(save_dir / "Bill_of_Rights_overview.txt", clean(html), url)


# ════════════════════════════════════════════════════════════════════════════
# UN / INTERNATIONAL (alternate sources for blocked OHCHR)
# ════════════════════════════════════════════════════════════════════════════

def download_international():
    info("═══ INTERNATIONAL HUMAN RIGHTS (alternate sources) ═══")
    save_dir = BASE_DIR / "international"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Use UN.org, Wikisource, and other mirrors that don't block scrapers
    docs = [
        ("https://www.un.org/en/about-us/universal-declaration-of-human-rights",
         "UDHR_Universal_Declaration_Human_Rights.txt"),
        ("https://www.un.org/en/genocideprevention/documents/atrocity-crimes/Doc.2_Convention%20on%20the%20Prevention%20and%20Punishment%20of%20the%20Crime%20of%20Genocide.pdf",
         "UN_Genocide_Convention.txt"),
        # Use law.cornell.edu for treaty texts they host
        ("https://www.law.cornell.edu/wex/international_covenant_on_civil_and_political_rights_(iccpr)",
         "ICCPR_Overview_Cornell.txt"),
        ("https://www.law.cornell.edu/wex/convention_against_torture",
         "Convention_Against_Torture_Overview_Cornell.txt"),
        ("https://www.law.cornell.edu/wex/international_human_rights",
         "International_Human_Rights_Overview_Cornell.txt"),
        ("https://www.law.cornell.edu/wex/geneva_conventions",
         "Geneva_Conventions_Overview_Cornell.txt"),
        # Use Justia for additional texts
        ("https://law.justia.com/constitution/us/amendment-14/",
         "14th_Amendment_Justia_Detailed.txt"),
        # Use University of Minnesota Human Rights Library
        ("http://hrlibrary.umn.edu/instree/z1afchpr.htm",
         "African_Charter_Human_Rights.txt"),
        ("http://hrlibrary.umn.edu/instree/auoam.htm",
         "American_Convention_Human_Rights.txt"),
        ("http://hrlibrary.umn.edu/instree/h2pasnhr.htm",
         "UN_Principles_Arrest_Detention.txt"),
        ("http://hrlibrary.umn.edu/instree/i1LAwenf.htm",
         "UN_Code_Conduct_Law_Enforcement.txt"),
        ("http://hrlibrary.umn.edu/instree/i2bpuff.htm",
         "UN_Basic_Principles_Use_of_Force_Law_Enforcement.txt"),
        ("http://hrlibrary.umn.edu/instree/g1LSmr.htm",
         "UN_Standard_Minimum_Rules_Prisoners.txt"),
        ("http://hrlibrary.umn.edu/instree/auoam.htm",
         "OAS_American_Convention_Human_Rights.txt"),
    ]
    
    for url, filename in docs:
        info(f"  {filename.replace('.txt','').replace('_',' ')}")
        html = fetch(url)
        if html:
            text = clean(html)
            if len(text) > 100:
                save_text(save_dir / filename, text, url)


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS (fixed URLs)
# ════════════════════════════════════════════════════════════════════════════

def download_statistics():
    info("═══ STATISTICS & DATA (fixed URLs) ═══")
    save_dir = BASE_DIR / "statistics"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    pages = [
        # BJS — updated URL structure
        ("https://bjs.ojp.gov/topics/law-enforcement", "BJS_law_enforcement_topics.txt"),
        ("https://bjs.ojp.gov/topics/corrections", "BJS_corrections_topics.txt"),
        ("https://bjs.ojp.gov/topics/courts-sentencing", "BJS_courts_sentencing.txt"),
        ("https://bjs.ojp.gov/topics/crime", "BJS_crime_topics.txt"),
        # FBI Crime Data Explorer
        ("https://cde.ucr.cjis.gov/", "FBI_Crime_Data_Explorer.txt"),
        # DOJ — these worked in v1
        ("https://www.justice.gov/crt/special-litigation-section-cases-and-matters",
         "DOJ_civil_rights_cases.txt"),
        ("https://www.justice.gov/crt/law-enforcement-misconduct",
         "DOJ_law_enforcement_misconduct.txt"),
        ("https://www.justice.gov/crt/addressing-police-misconduct-laws-enforced-department-justice",
         "DOJ_police_misconduct_laws.txt"),
        # Sentencing Project
        ("https://www.sentencingproject.org/research/",
         "Sentencing_Project_research.txt"),
        # Police Scorecard
        ("https://policescorecard.org/",
         "Police_Scorecard.txt"),
        # CT specific — OPM Criminal Justice Policy
        ("https://portal.ct.gov/OPM/CJ-About/CJ-About/Criminal-Justice-Policy-and-Planning-Division",
         "CT_Criminal_Justice_Policy.txt"),
        # Fatal Encounters
        ("https://fatalencounters.org/",
         "Fatal_Encounters.txt"),
        # Washington Post police shootings
        ("https://www.washingtonpost.com/graphics/investigations/police-shootings-database/",
         "WaPo_Police_Shootings_Database.txt"),
    ]
    
    for url, filename in pages:
        info(f"  {filename.replace('.txt','').replace('_',' ')}")
        html = fetch(url)
        if html:
            text = clean(html)
            if len(text) > 100:
                save_text(save_dir / filename, text, url)


# ════════════════════════════════════════════════════════════════════════════
# CT PRACTICE BOOK (Court Rules)
# ════════════════════════════════════════════════════════════════════════════

def download_ct_court_rules():
    info("═══ CT PRACTICE BOOK (Court Rules) ═══")
    save_dir = BASE_DIR / "state_statutes" / "ct_practice_book"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Key Practice Book chapters for criminal defense
    chapters = [
        ("https://www.jud.ct.gov/Publications/PracticeBook/PB_36.pdf", "PB_Ch36_Superior_Court_Criminal.pdf"),
        ("https://www.jud.ct.gov/Publications/PracticeBook/PB_37.pdf", "PB_Ch37_Arraignment.pdf"),
        ("https://www.jud.ct.gov/Publications/PracticeBook/PB_38.pdf", "PB_Ch38_Discovery.pdf"),
        ("https://www.jud.ct.gov/Publications/PracticeBook/PB_39.pdf", "PB_Ch39_Pretrial_Motions.pdf"),
        ("https://www.jud.ct.gov/Publications/PracticeBook/PB_40.pdf", "PB_Ch40_Trial_Procedure.pdf"),
        ("https://www.jud.ct.gov/Publications/PracticeBook/PB_41.pdf", "PB_Ch41_Sentencing.pdf"),
        ("https://www.jud.ct.gov/Publications/PracticeBook/PB_42.pdf", "PB_Ch42_Judgments.pdf"),
        ("https://www.jud.ct.gov/Publications/PracticeBook/PB_43.pdf", "PB_Ch43_Post_Trial_Motions.pdf"),
        ("https://www.jud.ct.gov/Publications/PracticeBook/PB_60.pdf", "PB_Ch60_Rules_of_Appellate_Procedure.pdf"),
        ("https://www.jud.ct.gov/Publications/PracticeBook/PB_61.pdf", "PB_Ch61_Appellate_Jurisdiction.pdf"),
    ]
    
    for url, filename in chapters:
        info(f"  {filename}")
        try:
            time.sleep(RATE_LIMIT)
            r = client.get(url)
            r.raise_for_status()
            path = save_dir / filename
            with open(path, "wb") as f:
                f.write(r.content)
            global downloaded
            downloaded += 1
            print(f"    ✓ {filename}", flush=True)
        except Exception as e:
            error(f"  {url} — {e}")


# ════════════════════════════════════════════════════════════════════════════
# INGESTION
# ════════════════════════════════════════════════════════════════════════════

def ingest_all():
    info("═══ INGESTING INTO KNOWLEDGE BASE ═══")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from config import load_config
        from tools.knowledge_base import KnowledgeBaseTool
        
        config = load_config()
        kb = KnowledgeBaseTool(config)
        
        for d in [
            BASE_DIR / "federal_statutes",
            BASE_DIR / "state_statutes",
            BASE_DIR / "case_law",
            BASE_DIR / "international",
            BASE_DIR / "statistics",
        ]:
            if d.exists():
                info(f"  Ingesting: {d}")
                result = kb.ingest_directory(str(d), recursive=True)
                print(f"  {result}\n", flush=True)
        
        info("Ingestion complete!")
        print(kb.knowledge_stats(), flush=True)
    except Exception as e:
        error(f"Ingestion failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Bulk Legal Downloader v2 (Fixed URLs)")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--ct-statutes", action="store_true", help="CT General Statutes (chapters + key sections)")
    parser.add_argument("--ct-rules", action="store_true", help="CT Practice Book (court rules PDFs)")
    parser.add_argument("--amendments", action="store_true", help="All Constitutional Amendments")
    parser.add_argument("--un", action="store_true", help="UN/International human rights (alternate sources)")
    parser.add_argument("--stats", action="store_true", help="Crime and justice statistics (fixed URLs)")
    parser.add_argument("--ingest", action="store_true", help="Ingest all downloads into knowledge base")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        print("\nExamples:")
        print("  python bulk_download_v2.py --all              # Everything")
        print("  python bulk_download_v2.py --ct-statutes      # CT statutes only")
        print("  python bulk_download_v2.py --amendments       # Constitutional amendments")
        print("  python bulk_download_v2.py --all --ingest     # Download + ingest")
        return
    
    start = datetime.now()
    print(f"\n{'═'*60}\n  BULK LEGAL DOWNLOADER v2\n  Output: {BASE_DIR}\n{'═'*60}\n")
    
    if args.all or args.ct_statutes:
        download_ct_statutes()
    if args.all or args.ct_rules:
        download_ct_court_rules()
    if args.all or args.amendments:
        download_amendments()
    if args.all or args.un:
        download_international()
    if args.all or args.stats:
        download_statistics()
    
    elapsed = datetime.now() - start
    print(f"\n{'═'*60}\n  COMPLETE\n  Downloaded: {downloaded} files\n  Errors: {errors}\n  Time: {elapsed}\n{'═'*60}\n")
    
    if args.ingest:
        ingest_all()
    else:
        print("To ingest: python bulk_download_v2.py --ingest\n")

if __name__ == "__main__":
    main()
