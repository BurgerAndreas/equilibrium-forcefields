

# find . -type d -iname '*E2MD17alphadrop01numlayers?pathdrop005*' -depth
# To replace all occurrences, use ${parameter//pattern/string}

# layers=(1 2 3 4 5 6 7 8)
# for layer in ${layers[@]}; do
#     find . -type d -iname '*E2MD17alphadrop01numlayers?pathdrop005*' -depth -exec bash -c '
#         echo mv "$1" "${1//E2MD17alphadrop01numlayers?pathdrop005/E2MD17ddnumlayers${layer}}"
#     ' -- {} \;
# done


# find . -type d -iname 'E2MD17*E2MD17dd*' -depth -exec bash -c '
#     echo mv "$1" "${1//E2MD17dd/dd}"
# ' -- {} \;


find . -type d -iname 'E2MD17*numlayers4*' -depth -exec bash -c '
    mv "$1" "${1//numlayers4/}"
' -- {} \;

# 
find . -type d -iname '*E2MD17*alphadrop01numlayers1pathdrop005*' -depth -exec bash -c '
    mv "$1" "${1//alphadrop01numlayers1pathdrop005/ddnumlayers1}"
' -- {} \;
# 
find . -type d -iname '*E2MD17*alphadrop01numlayers2pathdrop005*' -depth -exec bash -c '
    mv "$1" "${1//alphadrop01numlayers2pathdrop005/ddnumlayers2}"
' -- {} \;
# 
find . -type d -iname '*E2MD17*alphadrop01numlayers3pathdrop005*' -depth -exec bash -c '
    mv "$1" "${1//alphadrop01numlayers3pathdrop005/ddnumlayers3}"
' -- {} \;
# 
find . -type d -iname '*E2MD17*alphadrop01pathdrop005*' -depth -exec bash -c '
    mv "$1" "${1//alphadrop01pathdrop005/dd}"
' -- {} \;
# 
find . -type d -iname '*E2MD17*alphadrop01numlayers5pathdrop005*' -depth -exec bash -c '
    mv "$1" "${1//alphadrop01numlayers5pathdrop005/ddnumlayers5}"
' -- {} \;
# 
find . -type d -iname '*E2MD17*alphadrop01numlayers6pathdrop005*' -depth -exec bash -c '
    mv "$1" "${1//alphadrop01numlayers6pathdrop005/ddnumlayers6}"
' -- {} \;
# 
find . -type d -iname '*E2MD17*alphadrop01numlayers7pathdrop005*' -depth -exec bash -c '
    mv "$1" "${1//alphadrop01numlayers7pathdrop005/ddnumlayers7}"
' -- {} \;
# 
find . -type d -iname '*E2MD17*alphadrop01numlayers8pathdrop005*' -depth -exec bash -c '
    mv "$1" "${1//alphadrop01numlayers8pathdrop005/ddnumlayers8}"
' -- {} \;

# mv -f md17/equiformer_v2_md17/ethanol/ethanolweightdecay00 md17/equiformer_v2_md17/ethanol/E2MD17alphadrop01numlayers4pathdrop005ethanolweightdecay00
# ls models/md17/equiformer_v2_md17/ethanol