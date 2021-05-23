while getopts "d:e:" flag; do
        case $flag in
                d) directory=$OPTARG;;
        esac
done


if [ "$#" -lt 1 ]; then
        echo "Please check the parameters (-d)"
        echo "Usage: $0 -d [directory_name]"
        exit 1
fi
args=("$@")


perl ./tokenizer.perl -l de < ./../../log/$directory/test/output.txt > ./../../log/$directory/test/output
perl ./tokenizer.perl -l de < ./../../log/$directory/test/label.txt > ./../../log/$directory/test/label
perl ./multi-bleu.perl ./../../log/$directory/test/label < ./../../log/$directory/test/output
