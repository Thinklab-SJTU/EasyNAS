random=$(($RANDOM%100))

echo ${random}
if [ $random -gt 50 ]; then
    echo "acc: $random" 
elif [ $random -gt 30 ]; then
    echo "acc: $random" 
    exit 1
else
    exit 2
fi

