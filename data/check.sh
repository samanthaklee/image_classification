for president in trump obama bush clinton
do
    for n in 25 50 100 150
    do
        echo "$president $n"
        ls presidents_$n/$president | wc -l
        echo
    done
done
