with open('cv_parser_script/cv_parser8.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re
content = re.sub(r'",,', '",', content)

with open('cv_parser_script/cv_parser8.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Fixed double commas")
