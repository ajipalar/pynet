echo $1
[[ -f $1 ]] || (echo "Not a file" && exit 1)
[[ $1 = *.py ]] || (echo "Not a python file" && exit 1)
cat $1 | grep def | sed 's/def/  /' | sed 's/ */  /'

