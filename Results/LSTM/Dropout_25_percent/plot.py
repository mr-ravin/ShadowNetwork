import matplotlib.pyplot as plt
def valueret(filename):
  s=open(filename,"r")
  sdata=s.readlines()[0]
  #print(sdata)
  sdata_list=sdata.split(" ")
  s_val=[]
  count=0
  for i in sdata_list:
    s_val.append(float(i))
    count=count+1
  return (s_val,count)

s_val,count=valueret("shadow.txt")
u_val,count=valueret("uni.txt")
c_list=[]
for i in range(1,count+1):
  c_list.append(i)
plt.plot(c_list,s_val,label="LSTM with Shadow Network")
plt.plot(c_list,u_val,label="LSTM without Shadow Network")
plt.xlabel("Training Steps")
plt.ylabel("Training loss")
plt.legend()
plt.savefig("Test_LSTM.png")
plt.show()

