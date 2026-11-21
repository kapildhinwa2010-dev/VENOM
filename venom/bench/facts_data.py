"""The generated part of the fact bank, in a parseable form.

Format: items separated by ``;``, fields inside an item separated by ``/``.
Spaces inside a value are preserved, so "United Arab Emirates/Abu Dhabi" and
"Buenas Aires/Buenos Aires" both survive intact. An earlier version split on
whitespace, which silently broke every multi-word value.

The categories are the ones a sub-1B model has a non-zero chance on. Asking a
124M model about quantum chromodynamics produces a benchmark where the
baseline is 0% and every number is noise.
"""
from __future__ import annotations

from typing import Dict, List

CAPITALS = """
Afghanistan/Kabul; Albania/Tirana; Algeria/Algiers; Andorra/Andorra la Vella;
Angola/Luanda; Argentina/Buenos Aires; Armenia/Yerevan; Australia/Canberra;
Austria/Vienna; Azerbaijan/Baku; Bahamas/Nassau; Bahrain/Manama; Bangladesh/Dhaka;
Barbados/Bridgetown; Belarus/Minsk; Belgium/Brussels; Belize/Belmopan; Benin/Porto-Novo;
Bhutan/Thimphu; Bolivia/Sucre; Bosnia and Herzegovina/Sarajevo; Botswana/Gaborone;
Brazil/Brasilia; Brunei/Bandar Seri Begawan; Bulgaria/Sofia; Burkina Faso/Ouagadougou;
Burundi/Gitega; Cambodia/Phnom Penh; Cameroon/Yaounde; Canada/Ottawa; Cape Verde/Praia;
Central African Republic/Bangui; Chad/N'Djamena; Chile/Santiago; China/Beijing;
Colombia/Bogota; Comoros/Moroni; Congo/Brazzaville; Congo/Kinshasa; Costa Rica/San Jose;
Croatia/Zagreb; Cuba/Havana; Cyprus/Nicosia; Czech Republic/Prague; Denmark/Copenhagen;
Djibouti/Djibouti; Dominica/Roseau; Dominican Republic/Santo Domingo; Ecuador/Quito;
Egypt/Cairo; El Salvador/San Salvador; Equatorial Guinea/Malabo; Eritrea/Asmara;
Estonia/Tallinn; Eswatini/Mbabane; Ethiopia/Addis Ababa; Fiji/Suva; Finland/Helsinki;
France/Paris; Gabon/Libreville; Gambia/Banjul; Georgia/Tbilisi; Germany/Berlin;
Ghana/Accra; Greece/Athens; Grenada/St George's; Guatemala/Guatemala City; Guinea/Conakry;
Guinea-Bissau/Bissau; Guyana/Georgetown; Haiti/Port-au-Prince; Honduras/Tegucigalpa;
Hungary/Budapest; Iceland/Reykjavik; India/New Delhi; Indonesia/Jakarta; Iran/Tehran;
Iraq/Baghdad; Ireland/Dublin; Israel/Jerusalem; Italy/Rome; Jamaica/Kingston; Japan/Tokyo;
Jordan/Amman; Kazakhstan/Astana; Kenya/Nairobi; Kiribati/South Tarawa; Kuwait/Kuwait City;
Kyrgyzstan/Bishkek; Laos/Vientiane; Latvia/Riga; Lebanon/Beirut; Lesotho/Maseru;
Liberia/Monrovia; Libya/Tripoli; Liechtenstein/Vaduz; Lithuania/Vilnius; Luxembourg/Luxembourg;
Madagascar/Antananarivo; Malawi/Lilongwe; Malaysia/Kuala Lumpur; Maldives/Male;
Mali/Bamako; Malta/Valletta; Marshall Islands/Majuro; Mauritania/Nouakchott;
Mauritius/Port Louis; Mexico/Mexico City; Micronesia/Palikir; Moldova/Chisinau;
Monaco/Monaco; Mongolia/Ulaanbaatar; Montenegro/Podgorica; Morocco/Rabat; Mozambique/Maputo;
Myanmar/Naypyidaw; Namibia/Windhoek; Nepal/Kathmandu; Netherlands/Amsterdam;
New Zealand/Wellington; Nicaragua/Managua; Niger/Niamey; Nigeria/Abuja; North Macedonia/Skopje;
Norway/Oslo; Oman/Muscat; Pakistan/Islamabad; Palau/Ngerulmud; Panama/Panama City;
Papua New Guinea/Port Moresby; Paraguay/Asuncion; Peru/Lima; Philippines/Manila;
Poland/Warsaw; Portugal/Lisbon; Qatar/Doha; Romania/Bucharest; Russia/Moscow; Rwanda/Kigali;
Saint Lucia/Castries; Saint Vincent and the Grenadines/Kingstown; Samoa/Apia;
San Marino/San Marino; Sao Tome and Principe/Sao Tome; Saudi Arabia/Riyadh;
Senegal/Dakar; Serbia/Belgrade; Seychelles/Victoria; Sierra Leone/Freetown;
Singapore/Singapore; Slovakia/Bratislava; Slovenia/Ljubljana; Solomon Islands/Honiara;
Somalia/Mogadishu; South Africa/Pretoria; South Korea/Seoul; South Sudan/Juba;
Spain/Madrid; Sri Lanka/Sri Jayawardenepura Kotte; Sudan/Khartoum; Suriname/Paramaribo;
Sweden/Stockholm; Switzerland/Bern; Syria/Damascus; Tajikistan/Dushanbe; Tanzania/Dodoma;
Thailand/Bangkok; Timor-Leste/Dili; Togo/Lome; Tonga/Nuku'alofa; Tunisia/Tunis;
Turkey/Ankara; Turkmenistan/Ashgabat; Tuvalu/Funafuti; Uganda/Kampala; Ukraine/Kyiv;
United Arab Emirates/Abu Dhabi; United Kingdom/London; United States/Washington;
Uruguay/Montevideo; Uzbekistan/Tashkent; Vanuatu/Port Vila; Venezuela/Caracas;
Vietnam/Hanoi; Yemen/Sanaa; Zambia/Lusaka; Zimbabwe/Harare
"""

CURRENCIES = """
Japan/Yen; China/Yuan; India/Rupee; Russia/Ruble; United Kingdom/Pound;
Switzerland/Franc; Canada/Dollar; Australia/Dollar; Mexico/Peso; Brazil/Real;
South Korea/Won; Turkey/Lira; Indonesia/Rupiah; Sweden/Krona; Norway/Krone;
Denmark/Krone; Poland/Zloty; Czech Republic/Koruna; Hungary/Forint; Romania/Leu;
Bulgaria/Lev; Israel/Shekel; Saudi Arabia/Riyal; United Arab Emirates/Dirham;
Qatar/Riyal; Kuwait/Dinar; Nigeria/Naira; Kenya/Shilling; Ethiopia/Birr;
Ghana/Cedi; Egypt/Pound; Thailand/Baht; Vietnam/Dong; Philippines/Peso;
Pakistan/Rupee; Bangladesh/Taka; Sri Lanka/Rupee; Nepal/Rupee; Myanmar/Kyat;
Cambodia/Riel; New Zealand/Dollar; Singapore/Dollar; Hong Kong/Dollar; Taiwan/Dollar;
Argentina/Peso; Colombia/Peso; Chile/Peso; Peru/Sol; Venezuela/Bolivar;
Ukraine/Hryvnia; Serbia/Dinar; Kazakhstan/Tenge; Mongolia/Tugrik; Austria/Euro;
Belgium/Euro; Finland/Euro; Ireland/Euro; Portugal/Euro; Greece/Euro;
Slovakia/Euro; Slovenia/Euro; Estonia/Euro; Lithuania/Euro
"""

LANGUAGES = """
Brazil/Portuguese; Argentina/Spanish; Mexico/Spanish; Spain/Spanish;
France/French; Germany/German; Italy/Italian; Russia/Russian; China/Mandarin;
Japan/Japanese; Korea/Korean; India/Hindi; Pakistan/Urdu; Bangladesh/Bengali;
Nepali/Nepali; Indonesia/Indonesian; Malaysia/Malay; Thailand/Thai;
Vietnam/Vietnamese; Philippines/Filipino; Turkey/Turkish; Greece/Greek;
Persian/Persian; Arabic/Arabic; Swahili/Swahili; Dutch/Dutch; Swedish/Swedish;
Norwegian/Norwegian; Danish/Danish; Finnish/Finnish; Polish/Polish;
Czech/Czech; Hungarian/Hungarian; Romanian/Romanian; Bulgarian/Bulgarian;
Ukrainian/Ukrainian; Irish/Irish; Welsh/Welsh; Icelandic/Icelandic;
Estonian/Estonian; Latvian/Latvian; Lithuanian/Lithuanian; Catalan/Catalan
"""

ELEMENTS = """
H/Hydrogen/1; He/Helium/2; Li/Lithium/3; Be/Beryllium/4; B/Boron/5; C/Carbon/6;
N/Nitrogen/7; O/Oxygen/8; F/Fluorine/9; Ne/Neon/10; Na/Sodium/11; Mg/Magnesium/12;
Al/Aluminium/13; Si/Silicon/14; P/Phosphorus/15; S/Sulfur/16; Cl/Chlorine/17;
Ar/Argon/18; K/Potassium/19; Ca/Calcium/20; Fe/Iron/26; Co/Cobalt/27;
Ni/Nickel/28; Cu/Copper/29; Zn/Zinc/30; Ag/Silver/47; Sn/Tin/50; Au/Gold/79;
Hg/Mercury/80; Pb/Lead/82; U/Uranium/92
"""

SCIENTISTS = """
Isaac Newton/Physics/1643; Albert Einstein/Physics/1879; Galileo Galilei/Physics/1564;
Marie Curie/Chemistry/1867; Charles Darwin/Biology/1809; Louis Pasteur/Chemistry/1822;
Nikola Tesla/Physics/1856; Thomas Edison/Physics/1847; Alexander Graham Bell/Physics/1847;
Julius Caesar/History/100 BCE; Cleopatra/History/69 BCE; Joan of Arc/History/1412;
Max Planck/Physics/1858; Niels Bohr/Physics/1885; Enrico Fermi/Physics/1901;
Erwin Schrodinger/Physics/1887; Richard Feynman/Physics/1918; Stephen Hawking/Physics/1942;
Alan Turing/Computing/1912; Tim Berners-Lee/Computing/1955; Ada Lovelace/Computing/1815;
Charles Babbage/Computing/1791; Linus Torvalds/Computing/1969; Grace Hopper/Computing/1906;
Hedy Lamarr/Physics/1914; Rosalind Franklin/Chemistry/1920; Lise Meitner/Physics/1878;
Emmy Noether/Mathematics/1882; Ramanujan/Mathematics/1887; Gauss/Mathematics/1777;
Fibonacci/Mathematics/1170; Pythagoras/Mathematics/570 BCE; Archimedes/Mathematics/287 BCE;
Leonardo da Vinci/Art/1452; Michelangelo/Art/1475; Vincent van Gogh/Art/1853;
Claude Monet/Art/1840; Frida Kahlo/Art/1907; Rembrandt/Art/1606; Picasso/Art/1881;
Mozart/Music/1756; Beethoven/Music/1770; Bach/Music/1685
"""

BOOKS = """
Hamlet/William Shakespeare/1601; Macbeth/William Shakespeare/1606;
Romeo and Juliet/William Shakespeare/1595; Othello/William Shakespeare/1603;
Pride and Prejudice/Jane Austen/1813; Emma/Jane Austen/1815;
Sense and Sensibility/Jane Austen/1811; Jane Eyre/Charlotte Bronte/1847;
Wuthering Heights/Emily Bronte/1847; Great Expectations/Charles Dickens/1861;
Oliver Twist/Charles Dickens/1837; A Tale of Two Cities/Charles Dickens/1859;
David Copperfield/Charles Dickens/1850; Moby Dick/Herman Melville/1851;
War and Peace/Leo Tolstoy/1869; Anna Karenina/Leo Tolstoy/1878;
Crime and Punishment/Fyodor Dostoevsky/1866; Brothers Karamazov/Fyodor Dostoevsky/1880;
Les Miserables/Victor Hugo/1862; The Count of Monte Cristo/Alexandre Dumas/1844;
Madame Bovary/Gustave Flaubert/1857; Don Quixote/Miguel de Cervantes/1605;
The Odyssey/Homer/-700; The Iliad/Homer/-700; The Divine Comedy/Dante Alighieri/1320;
Paradise Lost/John Milton/1667; Gullivers Travels/Jonathan Swift/1726;
Robinson Crusoe/Daniel Defoe/1719; Frankenstein/Mary Shelley/1818; Dracula/Bram Stoker/1897;
Alice in Wonderland/Lewis Carroll/1865; Treasure Island/Robert Louis Stevenson/1883;
The Hobbit/J R R Tolkien/1937; 1984/George Orwell/1949; Animal Farm/George Orwell/1945;
Brave New World/Aldous Huxley/1932; To Kill a Mockingbird/Harper Lee/1960;
The Great Gatsby/F Scott Fitzgerald/1925; The Catcher in the Rye/J D Salinger/1951;
One Hundred Years of Solitude/Gabriel Garcia Marquez/1967;
The Old Man and the Sea/Ernest Hemingway/1952; The Trial/Franz Kafka/1925;
Metamorphosis/Franz Kafka/1915; Lord of the Flies/William Golding/1954;
The Handmaids Tale/Margaret Atwood/1985; Beloved/Toni Morrison/1987
"""

MISC = """
How many days are in a leap year/366/How many days are in a leap year;
How many continents are there/7/How many continents are there;
How many players are on a football team on the field/11/How many players are on a football team on the field;
How many sides does a hexagon have/6/How many sides does a hexagon have;
How many degrees are in a triangle/180/How many degrees are in a triangle;
How many bones are in the adult human body/206/How many bones are in the adult human body;
How many chambers does the human heart have/4/How many chambers does the human heart have;
How many strings does a standard guitar have/6/How many strings does a standard guitar have;
How many keys does a standard piano have/88/How many keys does a standard piano have;
How many ounces are in a pound/16/How many ounces are in a pound;
How many centimetres are in a metre/100/How many centimetres are in a metre;
How many days are in a week/7/How many days are in a week;
How many hours are in a day/24/How many hours are in a day;
How many minutes are in an hour/60/How many minutes are in an hour;
How many months have 31 days/7/How many months have 31 days;
How many bits are in a byte/8/How many bits are in a byte;
How many colours are in a rainbow/7/How many colours are in a rainbow;
What is the chemical symbol for gold/Au/chemical symbol for gold;
What is the chemical symbol for iron/Fe/chemical symbol for iron;
What is the chemical symbol for oxygen/O/chemical symbol for oxygen;
What is the chemical symbol for sodium/Na/chemical symbol for sodium;
What is the chemical symbol for silver/Ag/chemical symbol for silver;
What is the chemical symbol for carbon/C/chemical symbol for carbon;
What gas do plants absorb from the atmosphere/Carbon dioxide/gas plants absorb from the atmosphere;
What gas do humans exhale/Carbon dioxide/gas humans exhale;
What is the freezing point of water in Celsius/0/freezing point of water in Celsius;
What is the boiling point of water in Celsius/100/boiling point of water in Celsius;
What is the largest planet in the solar system/Jupiter/largest planet in the solar system;
What is the smallest planet in the solar system/Mercury/smallest planet in the solar system;
What is the closest star to Earth/The Sun/closest star to Earth;
What is the largest ocean in the world/The Pacific Ocean/largest ocean in the world;
What is the longest river in the world/The Nile/longest river in the world;
What is the tallest mountain in the world/Mount Everest/tallest mountain in the world;
What is the largest desert in the world/The Sahara/largest desert in the world;
What is the largest country in the world by land area/Russia/largest country in the world by land area;
What is the most populous country in the world/India/most populous country in the world;
What is the smallest country in the world/Vatican City/smallest country in the world;
Which planet is known as the Red Planet/Mars/planet known as the Red Planet;
What is the hardest natural substance/Diamond/hardest natural substance;
How many rings are on the Olympic flag/Five/Olympic flag rings;
How many squares are on a chessboard/64/chessboard squares;
How many players are on a basketball team on the court/Five/basketball team players on the court;
Who painted the Mona Lisa/Leonardo da Vinci/painter of the Mona Lisa;
Who wrote the play Romeo and Juliet/William Shakespeare/writer of Romeo and Juliet;
Who developed the theory of general relativity/Albert Einstein/developer of general relativity;
Who was the first person to walk on the moon/Neil Armstrong/first person to walk on the moon;
Who invented the telephone/Alexander Graham Bell/inventor of the telephone;
Who invented the electric light bulb/Thomas Edison/inventor of the electric light bulb;
Who formulated the laws of motion/Isaac Newton/formulator of the laws of motion;
Who discovered penicillin/Alexander Fleming/discoverer of penicillin;
Who was the first president of the United States/George Washington/first president of the United States;
In which year did the Berlin Wall fall/1989/year the Berlin Wall fell;
In which year did the Second World War end/1945/year the Second World War ended;
In which year did humans first land on the moon/1969/year humans first landed on the moon;
In which year was the United Nations founded/1945/year the United Nations was founded;
In which year was the Titanic launched/1912/year the Titanic was launched;
What is the speed of light in a vacuum/300000 kilometres per second/speed of light in a vacuum
"""


def _rows(blob: str) -> List[List[str]]:
    """Items separated by ';', fields inside an item separated by '/'."""
    rows = []
    for item in blob.split(";"):
        item = item.strip()
        if not item:
            continue
        fields = [f.strip() for f in item.split("/")]
        if len(fields) >= 2:
            rows.append(fields)
    return rows


def generated_facts() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    for country, capital in _rows(CAPITALS):
        out.append({"q": f"What is the capital of {country}?",
                    "gold": capital, "key": f"capital of {country.lower()}"})

    for country, currency in _rows(CURRENCIES):
        out.append({"q": f"What is the currency of {country}?",
                    "gold": currency, "key": f"currency of {country.lower()}"})

    for country, language in _rows(LANGUAGES):
        out.append({"q": f"What is the main language spoken in {country}?",
                    "gold": language, "key": f"language of {country.lower()}"})

    for symbol, name, number in _rows(ELEMENTS):
        out.append({"q": f"What is the chemical symbol for {name}?",
                    "gold": symbol, "key": f"chemical symbol for {name.lower()}"})
        out.append({"q": f"What is the atomic number of {name}?",
                    "gold": number, "key": f"atomic number of {name.lower()}"})

    for name, field, year in _rows(SCIENTISTS):
        out.append({"q": f"In which field did {name} work?",
                    "gold": field, "key": f"field of {name.lower()}"})
        out.append({"q": f"In which year was {name} born?",
                    "gold": year, "key": f"birth year of {name.lower()}"})

    for title, author, year in _rows(BOOKS):
        out.append({"q": f"Who wrote the book {title}?",
                    "gold": author, "key": f"author of {title.lower()}"})
        out.append({"q": f"In which year was the book {title} published?",
                    "gold": year, "key": f"publication year of {title.lower()}"})

    for fields in _rows(MISC):
        # a value may itself contain a slash ("300000 km/s"), so the question
        # is the first field, the answer the second, and the key the last
        out.append({"q": fields[0], "gold": fields[1], "key": fields[-1]})

    return out
