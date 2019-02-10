import requests
from bs4 import BeautifulSoup as bs
import os
import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

n_pages=100
n_captions=14
save_path='memes/'

df=pd.DataFrame(columns=['img_id','caption'])

for i in range(1,n_pages):
  if(i==1):
    url = 'https://memegenerator.net/memes/popular/alltime/'
  else:
    url = 'https://memegenerator.net/memes/popular/alltime/page/'+str(i)
    print("page_all:",str(i))

  r = requests.get(url)
  soup = bs(r.text,'html.parser')
  chars = soup.find_all(class_='char-img')
  imgs = [char.find('img') for char in chars]
  links = [char.find('a') for char in chars]

  for j,img in enumerate(imgs):
    img_url = img['src']
    response = requests.get(img_url,stream=True)
    name = img_url.split('/')[-1]
    complete_name = os.path.join(save_path, name)
    with open(complete_name,'wb') as im_file:
      im_file.write(response.content)
    del response

    for k in range(1,n_captions):
      print(links[j]['href'])
      if k==1:
        page_url = 'https://memegenerator.net' + links[j]['href']
      else:
        page_url = 'https://memegenerator.net' + links[j]['href'] + '/images/popular/alltime/page/' + str(k)
        print("page:",str(k))

      R = requests.get(page_url)
      s = bs(R.text,'html.parser')
      caps = s.find_all(class_='char-img')
      text0 = [cap.find(class_='optimized-instance-text0') for cap in caps]
      text1 = [cap.find(class_='optimized-instance-text1') for cap in caps]
      print(len(text0))
      assert len(text0) == len(text1)
      for t in range(len(text0)):
        df = df.append({'img_id':links[j]['href'],'caption':text0[t].text+"-"+text1[t].text},
                        ignore_index=True)
        
df.to_csv('data.csv')

