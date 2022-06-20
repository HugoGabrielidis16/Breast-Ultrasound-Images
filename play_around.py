import re


string = "tesstt_mask_1.png"
string2 = "mask_"
m = re.search(r"mask_[0-9]", string2)


if m:
    print(m.group(0))
