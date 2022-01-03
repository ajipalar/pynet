#Line							OFF	ON
assert 1==1    """assert 1==1"""		       #OK      OK	
#assert 1==2   """#assert 1==2  """                    #OK      OK
x = 2  #assert 1==3  """x = 2  #assert 1==3 """        #N       OK
# assert 1==4  """# assert 1==4 """                    #OK      OK
## assert 1==5 """## assert 1==5"""                    #OK      OK 
##assert 1==6  """##assert 1==6"""                     #OK      OK
#Empty comment """#Empty comment"""                    #OK      OK
##Double       """##Doube       """                    #OK      OK
def my_func(x):					       #OK      OK 
    assert x > 0				       #OK      OK
    return 2 / x				       #OK      OK
						       #OK      OK
if 1==1: assert 13 > 14 """if 1==1: assert 13 > 14"""
