#!/usr/bin/env python3
"""
BULK LEGAL DOWNLOADER
=====================
Downloads and organizes legal materials from free public sources.

Sources:
  - Connecticut General Statutes (cga.ct.gov)
  - Connecticut case law (CourtListener)
  - Connecticut General Assembly bills + public hearings (cga.ct.gov)
  - Federal US Code key titles (Cornell LII)
  - Federal case law (CourtListener)
  - UN / International human rights treaties (ohchr.org)
  - Statistics (BJS, FBI, CT OPM)

Usage:
  python bulk_download.py --all              # Download everything
  python bulk_download.py --ct               # Connecticut only
  python bulk_download.py --federal          # Federal only
  python bulk_download.py --international    # UN/International only
  python bulk_download.py --bills            # CT bills/hearings only
  python bulk_download.py --stats            # Statistics only
  python bulk_download.py --ingest           # Ingest everything after download

Everything saves to ~/LegalResearch/ organized by category.
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

# ── Config ──────────────────────────────────────────────────────────────────

BASE_DIR = Path.home() / "LegalResearch"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
}
RATE_LIMIT = 1.5  # seconds between requests — be polite

client = httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30, verify=False)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Track progress
downloaded = 0
errors = 0
skipped = 0


def info(msg): print(f"\033[0;32m[+]\033[0m {msg}", flush=True)
def warn(msg): print(f"\033[1;33m[!]\033[0m {msg}", flush=True)
def error(msg): print(f"\033[0;31m[✗]\033[0m {msg}", flush=True)


def save_text(path: Path, content: str, source_url: str = ""):
    """Save text content to file with metadata header."""
    global downloaded
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Source: {source_url}\n")
        f.write(f"# Downloaded: {datetime.now().isoformat()}\n\n")
        f.write(content)
    downloaded += 1


def fetch_page(url: str) -> str | None:
    """Fetch a web page with rate limiting."""
    global errors
    time.sleep(RATE_LIMIT)
    try:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        error(f"Failed: {url} — {e}")
        errors += 1
        return None


def extract_text(html: str) -> str:
    """Extract clean text from HTML."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.find("div", id="content") or soup
    text = main.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# CONNECTICUT STATUTES
# ════════════════════════════════════════════════════════════════════════════

CT_STATUTE_TITLES = {
    "1": "Provisions of General Application",
    "2": "State Jurisdiction and Boundaries",
    "3": "State Elective Officers",
    "4": "Management of State Agencies",
    "5": "State Employees",
    "7": "Municipalities",
    "8": "Zoning, Planning and Housing",
    "9": "Elections",
    "10": "Education and Culture",
    "12": "Taxation",
    "14": "Motor Vehicles, Use of the Highway",
    "17a": "Social and Human Services and Resources",
    "17b": "Social Services",
    "19a": "Public Health and Well-Being",
    "20": "Professional and Occupational Licensing",
    "21a": "Consumer Protection",
    "22a": "Environmental Protection",
    "29": "Public Safety and State Police",
    "30": "Intoxicating Liquors",
    "31": "Labor",
    "36a": "The Banking Law of Connecticut",
    "38a": "Insurance",
    "42": "Business, Selling, Trading and Collection Practices",
    "45a": "Probate Courts and Procedure",
    "46a": "Human Rights",
    "46b": "Family Law",
    "47": "Land and Land Titles",
    "47a": "Landlord and Tenant",
    "49": "Mortgages and Liens",
    "51": "Courts",
    "52": "Civil Actions",
    "53": "Crimes",
    "53a": "Penal Code",
    "54": "Criminal Procedure",
}


def download_ct_statutes():
    """Download Connecticut General Statutes from cga.ct.gov."""
    info("═══ CONNECTICUT GENERAL STATUTES ═══")
    save_dir = BASE_DIR / "state_statutes" / "connecticut"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://www.cga.ct.gov/current/pub"
    
    for title_num, title_name in CT_STATUTE_TITLES.items():
        info(f"  Title {title_num}: {title_name}")
        
        # Try to get the title's table of contents
        url = f"{base_url}/title{title_num}.htm"
        html = fetch_page(url)
        
        if html:
            text = extract_text(html)
            save_path = save_dir / f"title_{title_num}_{title_name.replace(' ', '_')[:40]}.txt"
            save_text(save_path, text, url)
            
            # Try to get individual chapters within the title
            soup = BeautifulSoup(html, "lxml")
            chapter_links = soup.find_all("a", href=re.compile(r"chap_\d+"))
            
            for link in chapter_links[:30]:  # Cap at 30 chapters per title
                chap_url = f"{base_url}/{link['href']}" if not link['href'].startswith('http') else link['href']
                chap_name = link.get_text(strip=True)[:60]
                
                chap_html = fetch_page(chap_url)
                if chap_html:
                    chap_text = extract_text(chap_html)
                    safe_name = re.sub(r'[^\w\-]', '_', chap_name)[:50]
                    chap_path = save_dir / f"title{title_num}_chap_{safe_name}.txt"
                    save_text(chap_path, chap_text, chap_url)
        else:
            warn(f"  Could not fetch title {title_num}")
    
    # Key individual statutes — criminal, civil rights, police
    info("  Downloading key individual statutes...")
    key_sections = [
        "sec_46a-58",   # Deprivation of rights - hate crimes
        "sec_46a-64",   # Discriminatory housing practices  
        "sec_46a-64c",  # Discriminatory public accommodation
        "sec_46a-71",   # Discriminatory employment practices
        "sec_53a-3",    # Penal code definitions
        "sec_53a-8",    # Criminal liability
        "sec_54-1f",    # Racial profiling prohibition
        "sec_54-33a",   # Search and seizure
        "sec_54-33b",   # Search warrants
        "sec_54-33c",   # Arrest without warrant
        "sec_29-6a",    # State police - use of force reporting
        "sec_29-6d",    # Police body cameras
        "sec_7-282",    # Municipal police training
    ]
    
    for sec in key_sections:
        url = f"{base_url}/{sec}.htm"
        html = fetch_page(url)
        if html:
            text = extract_text(html)
            save_path = save_dir / f"{sec.replace('-', '_')}.txt"
            save_text(save_path, text, url)


def download_ct_case_law():
    """Download Connecticut case law from CourtListener."""
    info("═══ CONNECTICUT CASE LAW ═══")
    save_dir = BASE_DIR / "case_law" / "connecticut"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Search key topics
    topics = [
        "excessive force police Connecticut",
        "civil rights violation Connecticut",
        "qualified immunity Connecticut",
        "false arrest Connecticut",
        "search seizure Connecticut",
        "police misconduct Connecticut",
        "racial profiling Connecticut",
        "wrongful conviction Connecticut",
        "due process Connecticut",
        "first amendment Connecticut",
        "fourth amendment Connecticut state",
        "municipal liability Connecticut",
        "state police Connecticut use of force",
        "prison conditions Connecticut",
        "employment discrimination Connecticut",
        "housing discrimination Connecticut",
    ]
    
    api_url = "https://www.courtlistener.com/api/rest/v4/search/"
    
    for topic in topics:
        info(f"  Searching: {topic}")
        try:
            resp = client.get(api_url, params={
                "q": topic,
                "type": "o",
                "court": "conn connappct ctd",  # CT Supreme, Appellate, District
                "order_by": "score desc",
                "page_size": 15,
            })
            resp.raise_for_status()
            results = resp.json().get("results", [])
            
            for case in results:
                case_name = case.get("caseName", "unknown")
                date = case.get("dateFiled", "unknown")
                court = case.get("court", "unknown")
                snippet = re.sub(r'<[^>]+>', '', case.get("snippet", ""))
                citation = ", ".join(case.get("citation", []))
                abs_url = case.get("absolute_url", "")
                
                safe_name = re.sub(r'[^\w]', '_', case_name)[:60]
                save_path = save_dir / f"{safe_name}_{date[:4] if date else 'unknown'}.txt"
                
                if not save_path.exists():
                    content = (
                        f"Case: {case_name}\n"
                        f"Court: {court}\n"
                        f"Date: {date}\n"
                        f"Citation: {citation}\n"
                        f"URL: https://www.courtlistener.com{abs_url}\n\n"
                        f"Summary:\n{snippet}\n"
                    )
                    
                    # Try to fetch full opinion
                    if abs_url:
                        opinion_html = fetch_page(f"https://www.courtlistener.com{abs_url}")
                        if opinion_html:
                            opinion_text = extract_text(opinion_html)
                            if len(opinion_text) > 200:
                                content += f"\n{'═' * 60}\nFULL OPINION:\n{'═' * 60}\n\n{opinion_text}"
                    
                    save_text(save_path, content, f"https://www.courtlistener.com{abs_url}")
            
            time.sleep(RATE_LIMIT)
            
        except Exception as e:
            warn(f"  Search failed for '{topic}': {e}")


# ════════════════════════════════════════════════════════════════════════════
# CT GENERAL ASSEMBLY — BILLS, HEARINGS, HISTORY
# ════════════════════════════════════════════════════════════════════════════

def download_ct_bills():
    """Download current and recent CT General Assembly bills and public hearings."""
    info("═══ CT GENERAL ASSEMBLY — BILLS & HEARINGS ═══")
    save_dir = BASE_DIR / "state_statutes" / "ct_bills"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Current session bills — search by committee
    committees = [
        "JUD",   # Judiciary
        "PS",    # Public Safety
        "GAE",   # Government Administration and Elections
        "HUM",   # Human Services
        "LAB",   # Labor
        "ED",    # Education
    ]
    
    for year in [2025, 2026]:
        info(f"  Session: {year}")
        
        # Bill search
        for comm in committees:
            url = f"https://www.cga.ct.gov/asp/cgabillstatus/cgabillstatus.asp?selBillType=Bill&which_year={year}&bill_num=&selCommittee={comm}"
            html = fetch_page(url)
            if html:
                text = extract_text(html)
                save_path = save_dir / f"{year}_{comm}_bills.txt"
                save_text(save_path, text, url)
        
        # Public hearing schedules
        hearing_url = f"https://www.cga.ct.gov/asp/menu/CommDocList.asp?comm_code=JUD&doc_type=PH&bill_type=&which_year={year}"
        html = fetch_page(hearing_url)
        if html:
            text = extract_text(html)
            save_path = save_dir / f"{year}_judiciary_hearings.txt"
            save_text(save_path, text, hearing_url)
    
    # CT legislative history — key police reform bills
    info("  Downloading key CT police reform legislation...")
    reform_bills = [
        ("2020", "HB-6004", "Police Accountability Act"),
        ("2021", "SB-1083", "Implementation of Police Accountability"),
        ("2022", "HB-5443", "Use of Force Standards"),
        ("2023", "SB-1198", "Law Enforcement Standards"),
    ]
    
    for year, bill, name in reform_bills:
        url = f"https://www.cga.ct.gov/asp/cgabillstatus/cgabillstatus.asp?selBillType=Bill&which_year={year}&bill_num={bill.split('-')[1]}"
        html = fetch_page(url)
        if html:
            text = extract_text(html)
            save_path = save_dir / f"{year}_{bill}_{name.replace(' ', '_')}.txt"
            save_text(save_path, text, url)
    
    # CT Law Revision Commission reports
    url = "https://www.cga.ct.gov/lrc/"
    html = fetch_page(url)
    if html:
        text = extract_text(html)
        save_text(save_dir / "law_revision_commission.txt", text, url)


# ════════════════════════════════════════════════════════════════════════════
# FEDERAL STATUTES
# ════════════════════════════════════════════════════════════════════════════

FEDERAL_TITLES = {
    # Most relevant titles for civil rights / criminal law work
    "5":  ("Government Organization and Employees", [
        "552", "552a", "552b",  # FOIA, Privacy Act
    ]),
    "18": ("Crimes and Criminal Procedure", [
        "241", "242", "245", "249",  # Civil rights crimes
        "1001",  # False statements
        "1341", "1343",  # Mail/wire fraud
        "1961", "1962",  # RICO
        "2510", "2511",  # Wiretapping
        "3142",  # Bail Reform Act
        "3161",  # Speedy Trial Act
        "3553",  # Sentencing factors
        "3582",  # Sentence modification
    ]),
    "28": ("Judiciary and Judicial Procedure", [
        "1331",  # Federal question jurisdiction
        "1332",  # Diversity jurisdiction
        "1343",  # Civil rights jurisdiction
        "1983",  # (actually title 42, but cross-ref)
        "2241", "2254", "2255",  # Habeas corpus
    ]),
    "34": ("Crime Control and Law Enforcement", [
        "12601", "12602",  # Pattern or practice investigations
    ]),
    "42": ("The Public Health and Welfare", [
        "1981", "1982", "1983", "1985", "1986", "1988",  # Civil Rights Act
        "2000a", "2000e",  # Title II, Title VII
        "3601", "3604", "3605", "3617",  # Fair Housing Act
        "12101", "12102", "12112",  # ADA
        "14141",  # Pattern or practice (old numbering)
    ]),
}


def download_federal_statutes():
    """Download key federal statutes from Cornell LII."""
    info("═══ FEDERAL STATUTES (US CODE) ═══")
    save_dir = BASE_DIR / "federal_statutes"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for title, (title_name, sections) in FEDERAL_TITLES.items():
        info(f"  Title {title}: {title_name}")
        
        for section in sections:
            url = f"https://www.law.cornell.edu/uscode/text/{title}/{section}"
            html = fetch_page(url)
            
            if html:
                text = extract_text(html)
                if len(text) > 100:
                    save_path = save_dir / f"{title}_USC_{section}.txt"
                    save_text(save_path, text, url)
                    print(f"    ✓ {title} USC § {section}", flush=True)
                else:
                    print(f"    ⏭ {title} USC § {section} (no content)", flush=True)
            else:
                print(f"    ✗ {title} USC § {section} (failed)", flush=True)
    
    # Constitutional Amendments
    info("  Downloading Constitutional Amendments...")
    amendments = [
        ("1", "First Amendment — Free Speech, Religion, Assembly"),
        ("2", "Second Amendment — Right to Bear Arms"),
        ("4", "Fourth Amendment — Search and Seizure"),
        ("5", "Fifth Amendment — Due Process, Self-Incrimination"),
        ("6", "Sixth Amendment — Right to Counsel, Speedy Trial"),
        ("8", "Eighth Amendment — Cruel and Unusual Punishment"),
        ("13", "Thirteenth Amendment — Abolition of Slavery"),
        ("14", "Fourteenth Amendment — Equal Protection, Due Process"),
        ("15", "Fifteenth Amendment — Voting Rights (Race)"),
        ("19", "Nineteenth Amendment — Women's Suffrage"),
    ]
    
    for num, name in amendments:
        url = f"https://www.law.cornell.edu/constitution/amendmentxiv"  # Template
        # Use proper roman numeral mapping
        roman_map = {"1":"i","2":"ii","4":"iv","5":"v","6":"vi","8":"viii",
                     "13":"xiii","14":"xiv","15":"xv","19":"xix"}
        roman = roman_map.get(num, num)
        url = f"https://www.law.cornell.edu/constitution/amendment{roman}"
        
        html = fetch_page(url)
        if html:
            text = extract_text(html)
            save_path = save_dir / f"amendment_{num}_{name.split('—')[0].strip().replace(' ', '_')}.txt"
            save_text(save_path, text, url)
            print(f"    ✓ Amendment {num}", flush=True)
    
    # Federal Rules
    info("  Downloading Federal Rules...")
    rules = [
        ("https://www.law.cornell.edu/rules/fre", "federal_rules_of_evidence.txt"),
        ("https://www.law.cornell.edu/rules/frcp", "federal_rules_civil_procedure.txt"),
        ("https://www.law.cornell.edu/rules/frcrmp", "federal_rules_criminal_procedure.txt"),
    ]
    
    for url, filename in rules:
        html = fetch_page(url)
        if html:
            text = extract_text(html)
            save_text(save_dir / filename, text, url)


def download_federal_case_law():
    """Download landmark federal case law from CourtListener."""
    info("═══ FEDERAL CASE LAW ═══")
    save_dir = BASE_DIR / "case_law" / "federal"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Landmark cases by topic
    landmark_searches = [
        "Graham v. Connor excessive force",
        "Terry v. Ohio stop and frisk",
        "Tennessee v. Garner deadly force",
        "Monell v. Department of Social Services",
        "Monroe v. Pape section 1983",
        "Mapp v. Ohio exclusionary rule",
        "Miranda v. Arizona",
        "Gideon v. Wainwright right to counsel",
        "Brady v. Maryland disclosure",
        "Batson v. Kentucky jury discrimination",
        "Strickland v. Washington ineffective counsel",
        "Harlow v. Fitzgerald qualified immunity",
        "Pearson v. Callahan qualified immunity",
        "Saucier v. Katz qualified immunity",
        "Hope v. Pelzer clearly established",
        "Scott v. Harris vehicle pursuit",
        "Kingsley v. Hendrickson pretrial detainee",
        "Ashcroft v. Iqbal plausibility pleading",
        "Bell Atlantic v. Twombly pleading standard",
        "City of Canton v. Harris failure to train",
        "Los Angeles v. Lyons standing injunction",
        "Whren v. United States pretextual stop",
        "Illinois v. Wardlow flight reasonable suspicion",
        "Rodriguez v. United States traffic stop duration",
        "Carpenter v. United States cell phone privacy",
        "Riley v. California cell phone search",
    ]
    
    api_url = "https://www.courtlistener.com/api/rest/v4/search/"
    
    for query in landmark_searches:
        info(f"  {query}")
        try:
            resp = client.get(api_url, params={
                "q": query,
                "type": "o",
                "order_by": "score desc",
                "page_size": 3,
            })
            resp.raise_for_status()
            results = resp.json().get("results", [])
            
            for case in results[:2]:  # Top 2 per search
                case_name = case.get("caseName", "unknown")
                date = case.get("dateFiled", "")
                court = case.get("court", "")
                citation = ", ".join(case.get("citation", []))
                abs_url = case.get("absolute_url", "")
                snippet = re.sub(r'<[^>]+>', '', case.get("snippet", ""))
                
                safe_name = re.sub(r'[^\w]', '_', case_name)[:60]
                save_path = save_dir / f"{safe_name}.txt"
                
                if not save_path.exists():
                    content = (
                        f"Case: {case_name}\n"
                        f"Court: {court}\n"
                        f"Date: {date}\n"
                        f"Citation: {citation}\n"
                        f"URL: https://www.courtlistener.com{abs_url}\n\n"
                        f"Summary:\n{snippet}\n"
                    )
                    
                    if abs_url:
                        opinion_html = fetch_page(f"https://www.courtlistener.com{abs_url}")
                        if opinion_html:
                            opinion_text = extract_text(opinion_html)
                            if len(opinion_text) > 200:
                                content += f"\n{'═' * 60}\nFULL OPINION:\n{'═' * 60}\n\n{opinion_text}"
                    
                    save_text(save_path, content, f"https://www.courtlistener.com{abs_url}")
            
            time.sleep(RATE_LIMIT)
        except Exception as e:
            warn(f"  Failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# INTERNATIONAL / UN HUMAN RIGHTS
# ════════════════════════════════════════════════════════════════════════════

def download_international():
    """Download UN and international human rights instruments."""
    info("═══ INTERNATIONAL HUMAN RIGHTS ═══")
    save_dir = BASE_DIR / "international"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Core UN human rights instruments
    un_documents = [
        ("https://www.un.org/en/about-us/universal-declaration-of-human-rights",
         "UN_Universal_Declaration_Human_Rights.txt"),
        ("https://www.ohchr.org/en/instruments-mechanisms/instruments/international-covenant-civil-and-political-rights",
         "UN_ICCPR_Civil_Political_Rights.txt"),
        ("https://www.ohchr.org/en/instruments-mechanisms/instruments/international-covenant-economic-social-and-cultural-rights",
         "UN_ICESCR_Economic_Social_Cultural_Rights.txt"),
        ("https://www.ohchr.org/en/instruments-mechanisms/instruments/convention-against-torture-and-other-cruel-inhuman-or-degrading",
         "UN_Convention_Against_Torture.txt"),
        ("https://www.ohchr.org/en/instruments-mechanisms/instruments/international-convention-elimination-all-forms-racial",
         "UN_Convention_Racial_Discrimination.txt"),
        ("https://www.ohchr.org/en/instruments-mechanisms/instruments/convention-elimination-all-forms-discrimination-against-women",
         "UN_CEDAW_Discrimination_Against_Women.txt"),
        ("https://www.ohchr.org/en/instruments-mechanisms/instruments/convention-rights-child",
         "UN_Convention_Rights_of_Child.txt"),
        ("https://www.ohchr.org/en/instruments-mechanisms/instruments/international-convention-protection-rights-all-migrant",
         "UN_Convention_Migrant_Workers.txt"),
        ("https://www.ohchr.org/en/instruments-mechanisms/instruments/convention-rights-persons-disabilities",
         "UN_Convention_Persons_Disabilities.txt"),
        ("https://www.ohchr.org/en/instruments-mechanisms/instruments/international-convention-protection-all-persons-enforced",
         "UN_Convention_Enforced_Disappearance.txt"),
        # Policing specific
        ("https://www.ohchr.org/en/instruments-mechanisms/instruments/code-conduct-law-enforcement-officials",
         "UN_Code_Conduct_Law_Enforcement.txt"),
        ("https://www.ohchr.org/en/instruments-mechanisms/instruments/basic-principles-use-force-and-firearms-law-enforcement",
         "UN_Basic_Principles_Use_Force_Law_Enforcement.txt"),
        ("https://www.ohchr.org/en/instruments-mechanisms/instruments/body-principles-protection-all-persons-under-any-form",
         "UN_Principles_Protection_Detained_Persons.txt"),
        ("https://www.ohchr.org/en/instruments-mechanisms/instruments/standard-minimum-rules-treatment-prisoners",
         "UN_Standard_Minimum_Rules_Prisoners_Nelson_Mandela.txt"),
    ]
    
    for url, filename in un_documents:
        info(f"  {filename.replace('.txt','').replace('_',' ')}")
        html = fetch_page(url)
        if html:
            text = extract_text(html)
            save_text(save_dir / filename, text, url)
    
    # Inter-American Commission on Human Rights (relevant to US)
    iachr_docs = [
        ("https://www.oas.org/en/iachr/mandate/basics/declaration.asp",
         "American_Declaration_Rights_Duties_Man.txt"),
        ("https://www.oas.org/dil/inter_american_convention_to_prevent_and_punish_torture.htm",
         "InterAmerican_Convention_Prevent_Torture.txt"),
    ]
    
    for url, filename in iachr_docs:
        info(f"  {filename.replace('.txt','').replace('_',' ')}")
        html = fetch_page(url)
        if html:
            text = extract_text(html)
            save_text(save_dir / filename, text, url)
    
    # Geneva Conventions (summary)
    info("  Geneva Conventions...")
    url = "https://www.icrc.org/en/doc/war-and-law/treaties-customary-law/geneva-conventions/overview-geneva-conventions.htm"
    html = fetch_page(url)
    if html:
        text = extract_text(html)
        save_text(save_dir / "Geneva_Conventions_Overview.txt", text, url)


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ════════════════════════════════════════════════════════════════════════════

def download_statistics():
    """Download crime and justice statistics from government sources."""
    info("═══ STATISTICS & DATA ═══")
    save_dir = BASE_DIR / "statistics"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    stats_pages = [
        # Bureau of Justice Statistics
        ("https://bjs.ojp.gov/topics/law-enforcement/police-use-of-force",
         "BJS_police_use_of_force.txt"),
        ("https://bjs.ojp.gov/topics/law-enforcement/contacts-between-police-and-the-public",
         "BJS_police_public_contacts.txt"),
        ("https://bjs.ojp.gov/topics/courts-and-sentencing/sentencing",
         "BJS_sentencing.txt"),
        ("https://bjs.ojp.gov/topics/corrections/prisoners",
         "BJS_prisoners.txt"),
        ("https://bjs.ojp.gov/topics/civil-rights/hate-crime",
         "BJS_hate_crime.txt"),
        
        # FBI
        ("https://ucr.fbi.gov/crime-in-the-u.s",
         "FBI_UCR_crime_in_us.txt"),
        ("https://www.fbi.gov/how-can-we-help-you/more-fbi-services-and-information/ucr/use-of-force",
         "FBI_use_of_force_data.txt"),
        
        # DOJ Civil Rights Division
        ("https://www.justice.gov/crt/special-litigation-section-cases-and-matters",
         "DOJ_civil_rights_cases.txt"),
        ("https://www.justice.gov/crt/law-enforcement-misconduct",
         "DOJ_law_enforcement_misconduct.txt"),
        
        # CT specific
        ("https://www.ct.gov/despp/cwp/view.asp?a=4211&q=494782",
         "CT_use_of_force_reports.txt"),
        ("https://portal.ct.gov/POSTC",
         "CT_Police_Officer_Standards_Training.txt"),
    ]
    
    for url, filename in stats_pages:
        info(f"  {filename.replace('.txt','').replace('_',' ')}")
        html = fetch_page(url)
        if html:
            text = extract_text(html)
            save_text(save_dir / filename, text, url)
    
    # The Sentencing Project — racial disparities
    info("  Sentencing Project racial disparities...")
    url = "https://www.sentencingproject.org/reports/the-color-of-justice-racial-and-ethnic-disparity-in-state-prisons-the-sentencing-project/"
    html = fetch_page(url)
    if html:
        text = extract_text(html)
        save_text(save_dir / "sentencing_project_racial_disparity.txt", text, url)
    
    # Mapping Police Violence
    info("  Mapping Police Violence...")
    url = "https://mappingpoliceviolence.us/"
    html = fetch_page(url)
    if html:
        text = extract_text(html)
        save_text(save_dir / "mapping_police_violence.txt", text, url)


# ════════════════════════════════════════════════════════════════════════════
# INGESTION
# ════════════════════════════════════════════════════════════════════════════

def ingest_all():
    """Ingest all downloaded legal materials into the knowledge base."""
    info("═══ INGESTING INTO KNOWLEDGE BASE ═══")
    
    try:
        from config import load_config
        from tools.knowledge_base import KnowledgeBaseTool
        
        config = load_config()
        kb = KnowledgeBaseTool(config)
        
        dirs_to_ingest = [
            BASE_DIR / "federal_statutes",
            BASE_DIR / "state_statutes",
            BASE_DIR / "case_law",
            BASE_DIR / "international",
            BASE_DIR / "statistics",
            BASE_DIR / "news_clips",
        ]
        
        for d in dirs_to_ingest:
            if d.exists():
                info(f"  Ingesting: {d}")
                result = kb.ingest_directory(str(d), recursive=True)
                print(f"  {result}\n", flush=True)
            else:
                warn(f"  Skipping (not found): {d}")
        
        info("Ingestion complete!")
        stats = kb.knowledge_stats()
        print(stats, flush=True)
        
    except Exception as e:
        error(f"Ingestion failed: {e}")
        error("Run from the project directory: cd ~/Desktop/AxisAIAssistant/local-ai-assistant && python bulk_download.py --ingest")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Bulk Legal Downloader")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--ct", action="store_true", help="Connecticut statutes + case law")
    parser.add_argument("--federal", action="store_true", help="Federal statutes + case law")
    parser.add_argument("--international", action="store_true", help="UN/International human rights")
    parser.add_argument("--bills", action="store_true", help="CT General Assembly bills + hearings")
    parser.add_argument("--stats", action="store_true", help="Crime and justice statistics")
    parser.add_argument("--ingest", action="store_true", help="Ingest all downloads into knowledge base")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        print("\nExamples:")
        print("  python bulk_download.py --all              # Everything")
        print("  python bulk_download.py --ct --federal     # CT + Federal")
        print("  python bulk_download.py --all --ingest     # Download + ingest")
        return
    
    start = datetime.now()
    
    print(f"\n{'═' * 60}")
    print(f"  BULK LEGAL DOWNLOADER")
    print(f"  Output: {BASE_DIR}")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═' * 60}\n")
    
    if args.all or args.ct:
        download_ct_statutes()
        download_ct_case_law()
    
    if args.all or args.bills:
        download_ct_bills()
    
    if args.all or args.federal:
        download_federal_statutes()
        download_federal_case_law()
    
    if args.all or args.international:
        download_international()
    
    if args.all or args.stats:
        download_statistics()
    
    elapsed = datetime.now() - start
    
    print(f"\n{'═' * 60}")
    print(f"  COMPLETE")
    print(f"  Downloaded: {downloaded} files")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed}")
    print(f"  Location: {BASE_DIR}")
    print(f"{'═' * 60}\n")
    
    if args.ingest:
        ingest_all()
    else:
        print("To ingest into the knowledge base:")
        print(f"  python bulk_download.py --ingest\n")


if __name__ == "__main__":
    main()
