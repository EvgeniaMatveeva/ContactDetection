#!/usr/bin/env bash

# train data
wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1xGyiefcd_LtDUOWUzVWq6BRkEXugmlsC' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xGyiefcd_LtDUOWUzVWq6BRkEXugmlsC" \
-O train.tar.xz && rm -rf /tmp/cookies.txt && \
tar -xf train.tar.xz && rm -rf train.tar.xz

# val data
wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1rEwfbIlwKAPlAzIHP5oJHgo4bbRLi996' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rEwfbIlwKAPlAzIHP5oJHgo4bbRLi996" \
-O val.csv && rm -rf /tmp/cookies.txt