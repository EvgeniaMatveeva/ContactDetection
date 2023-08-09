#!/bin/bash

# default BERT
wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=12ertcocCmjkfghzKgVqFwxdjGVzoRiL0' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12ertcocCmjkfghzKgVqFwxdjGVzoRiL0" \
-O bert.pt && rm -rf /tmp/cookies.txt

# pretrained BERT
wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1Akp0i269sagCYdR0CUUo-GkB2__O8lZQ' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Akp0i269sagCYdR0CUUo-GkB2__O8lZQ" \
-O bert_pretrained.tar.xz && rm -rf /tmp/cookies.txt && \
tar -xf bert_pretrained.tar.xz && rm -rf bert_pretrained.tar.xz

# BERT tokenizer
wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1NTBPvFkTOM9cyI5dOxVxExsPXUCLgK7B' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NTBPvFkTOM9cyI5dOxVxExsPXUCLgK7B" \
-O tokenizer_pretrained.tar.xz && rm -rf /tmp/cookies.txt && \
tar -xf tokenizer_pretrained.tar.xz && rm -rf tokenizer_pretrained.tar.xz

# BERTs for categories
wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1VyN_qE0smgzobTf5jzgd0a6EZ62ZHADV' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VyN_qE0smgzobTf5jzgd0a6EZ62ZHADV" \
-O bert_cat0.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1w1s0_0ALWGcgq0rVyhWIHiyp2VWIcyav' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1w1s0_0ALWGcgq0rVyhWIHiyp2VWIcyav" \
-O bert_cat1.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1r2LhpTI56kNIUlBYHkamtqgDVupHhAP7' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1r2LhpTI56kNIUlBYHkamtqgDVupHhAP7" \
-O bert_cat2.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=19ZAUYDFcDRlNYnmJPC0eNWWWw8J1sBk7' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19ZAUYDFcDRlNYnmJPC0eNWWWw8J1sBk7" \
-O bert_cat3.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1NI_bH9TCT2RixOvwwmA2Qm6V5L4TUen8' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NI_bH9TCT2RixOvwwmA2Qm6V5L4TUen8" \
-O bert_cat4.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1clj-TMJ1IakOBSxY9_PlUQeG8xHPXuMN' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1clj-TMJ1IakOBSxY9_PlUQeG8xHPXuMN" \
-O bert_cat5.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1MgVMqWPo4zh6IhLUu3un18Acmg0Z0Ki1' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MgVMqWPo4zh6IhLUu3un18Acmg0Z0Ki1" \
-O bert_cat6.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1f3HeHyqzpkr6inB6heLpuVIo9hMpO4RV' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1f3HeHyqzpkr6inB6heLpuVIo9hMpO4RV" \
-O bert_cat7.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1wDTiP81VppJeBIPG99AtGedui-ugTjey' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wDTiP81VppJeBIPG99AtGedui-ugTjey" \
-O bert_cat8.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1b8A1QLxLH0LFKPSjwieRpmSXInpkztDP' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1b8A1QLxLH0LFKPSjwieRpmSXInpkztDP" \
-O bert_cat9.pt && rm -rf /tmp/cookies.txt

# Category to index dictionary
wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1SSmSAmqi0qr3qEJ3FA0Cr-77YiHkk5lR' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1SSmSAmqi0qr3qEJ3FA0Cr-77YiHkk5lR" \
-O cat2ind.pkl && rm -rf /tmp/cookies.txt