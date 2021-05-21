from selenium import webdriver 
from selenium.webdriver.common.keys import Keys 
import selenium.common.exceptions 
import time 
from bs4 import BeautifulSoup as bs 
import requests 
import os
from unicodedata import normalize 
import json
from collections import OrderedDict
import re
import time

username = input('Enter Your User Name ') 
password = input('Enter Your Password ') 
url = 'https://instagram.com/explore/tags/' + input('Enter Hashtag For Downloading Posts ')
start_date = int(input('Enter Start date YYYYMMDD   '))
end_date = int(input('Enter End date YYYYMMDD   '))

def path(): 
    global chrome 
    chrome = webdriver.Chrome('C:/Users/장수지/Downloads/chromedriver_win32 (1)/chromedriver.exe')
    
def url_name(url): 
    chrome.get(url) 
    time.sleep(10) 
      
def login(username, your_password): 
    try:
        log_but = chrome.find_element_by_class_name("L3NKy") 
        time.sleep(10) 
        log_but.click() 
        time.sleep(10)
    except:
        pass
    usern = chrome.find_element_by_name("username") 
    usern.send_keys(username) 
    passw = chrome.find_element_by_name("password") 
    passw.send_keys(your_password) 
    passw.send_keys(Keys.RETURN) 
  
    time.sleep(10) 
      
def first_post(): 
    pic = chrome.find_element_by_class_name("kIKUG").click() 
    time.sleep(2) 
      
def next_post(): 
    try: 
        nex = chrome.find_element_by_class_name( 
            "coreSpriteRightPaginationArrow") 
        return nex 
    except selenium.common.exceptions.NoSuchElementException: 
        return 0
        
def download_allposts(): 
    url_name(url)
    first_post() 
    user_name = url.split('/')[-1] 
    all_data = OrderedDict()
    if(os.path.isdir(user_name) == False): 
        os.mkdir(user_name) 
    multiple_images = nested_check() 
    elem_text = chrome.find_element_by_class_name('C4VMK').text
    elem_text = elem_text.replace('\n', ' ')
    elem_text = elem_text.split(' ')
    elem_text.remove(elem_text[0])
    elem_text = elem_text[:-1]
    elem_text = ' '.join(elem_text)
    elem_text = normalize('NFC',elem_text)
    tag = re.findall('#[A-Za-z0-9가-힣]+', elem_text)
    
    html = chrome.page_source
    soup = bs(html, 'lxml') 
    maxnum = int(chrome.find_element_by_class_name('g47SY').text.replace(',', ''))
    print('['+user_name+' 게시물 수]',maxnum)
    try: 
        date = str(soup.select('time._1o9PC.Nzb55')[0]['datetime'])[:10]
    except: 
        date = ''

    num_date = int(date.replace('-', ''))
    if start_date<=num_date<=end_date:
        file_data = OrderedDict()
        file_data["date"] = date
        file_data["tag"] = tag
        file_data["text"] = elem_text
        name_list = []
        if multiple_images: 
            nescheck = multiple_images 
            count_img = 0
            while nescheck: 
                elem_img = chrome.find_element_by_class_name('rQDP3') 
                save_multiple(user_name+'/'+'content1.'+str(count_img), elem_img) 
                name_list.append('content1.'+str(count_img)+'.jpg')
                count_img += 1
                nescheck.click() 
                nescheck = nested_check() 
            save_multiple(user_name+'/'+'content1.' +
                        str(count_img), elem_img, last_img_flag=1) 
            name_list.append('content1.'+str(count_img)+'.jpg')
            file_data["name"] = name_list
        else: 
            save_content('_97aPb', user_name+'/'+'content1') 
            name_list.append('content1'+'.jpg')
            file_data["name"] = name_list
        all_data['content1'] = file_data
    c = 2
      
    while(True): 
        next_el = next_post() 
        if next_el != False: 
            next_el.click() 
            time.sleep(1.3) 
              
            try:
                
                elem_text = chrome.find_element_by_class_name('C4VMK').text
                elem_text = elem_text.replace('\n', ' ')
                elem_text = elem_text.split(' ')
                elem_text.remove(elem_text[0])
                elem_text = elem_text[:-1]
                elem_text = ' '.join(elem_text)
                elem_text = normalize('NFC',elem_text)
                tag = re.findall('#[A-Za-z0-9가-힣]+', elem_text)
                try: 
                    html = chrome.page_source
                    soup = bs(html, 'lxml') 
                    date = str(soup.select('time._1o9PC.Nzb55')[0]['datetime'])[:10]
                except: 
                    date = ''

                num_date = int(date.replace('-', ''))
                if start_date<=num_date<=end_date:
                    file_data = OrderedDict()
                    file_data["date"] = date
                    file_data["tag"] = tag
                    file_data["text"] = elem_text
                    name_list = []

                    multiple_images = nested_check() 
                    if multiple_images: 
                        nescheck = multiple_images 
                        count_img = 0
                        
                        while nescheck: 
                            elem_img = chrome.find_element_by_class_name('rQDP3') 
                            save_multiple(user_name+'/'+'content' +
                                        str(c)+'.'+str(count_img), elem_img) 
                            name_list.append('content'+str(c)+'.'+str(count_img)+'.jpg')
                            count_img += 1
                            nescheck.click() 
                            nescheck = nested_check() 
                        save_multiple(user_name+'/'+'content'+str(c) +
                                    '.'+str(count_img), elem_img, 1) 
                        name_list.append('content'+str(c)+'.'+str(count_img)+'.jpg')
                        file_data["name"] = name_list
                    else: 
                        save_content('_97aPb', user_name+'/'+'content'+str(c)) 
                        name_list.append('content'+str(c)+'.jpg')
                        file_data["name"] = name_list
                    all_data['content'+str(c)] = file_data
                    with open(user_name+'/'+user_name+'.json','w', encoding='UTF8') as f:
                        json.dump(all_data, f, ensure_ascii=False, indent='\t')

            except (ConnectionError, TimeoutError):
                print("Trying to Reconnect...")
                time.sleep(10)
                continue
            except selenium.common.exceptions.NoSuchElementException: 
                print("Trying to Reconnect...")
                print("If there are no posts remaining, please press Ctrl+C")
                time.sleep(10)
                continue
            except:
                print("Unknown Error...")
                print("Trying to Reconnect...")
                time.sleep(10)
                continue                
        else: 
            break
        c += 1
        if maxnum<=c:
            print("finished") 
            return
    
    


def save_content(class_name, img_name): 
    time.sleep(0.5) 
    try: 
        pic = chrome.find_element_by_class_name(class_name) 
    except selenium.common.exceptions.NoSuchElementException: 
        print("Either This user has no images or you haven't followed this user or something went wrong") 
        return
    html = pic.get_attribute('innerHTML') 
    soup = bs(html, 'html.parser') 
    link = soup.find('video') 
    try:
        if link: 
            link = link['src'] 
          
        else: 
            link = soup.find('img')['src']
        response = requests.get(link)
    except:
        print("Trying to Reconnect...")
        time.sleep(10)
        return

    with open(img_name+'.jpg', 'wb') as f: 
        f.write(response.content)
    time.sleep(0.9) 
      
def save_multiple(img_name, elem, last_img_flag=False): 
    time.sleep(1) 
    l = elem.get_attribute('innerHTML') 
    html = bs(l, 'html.parser') 
    biglist = html.find_all('ul') 
    biglist = biglist[0] 
    list_images = biglist.find_all('li') 
      
    if last_img_flag: 
        user_image = list_images[-1] 
      
    else: 
        user_image = list_images[(len(list_images)//2)] 
    video = user_image.find('video') 
      
    if video: 
        return 
      
    else: 
        try:
            link = user_image.find('img')['src'] 
            response = requests.get(link) 
        except:
            time.sleep(10)
            return

    with open(img_name+'.jpg', 'wb') as f: 
        f.write(response.content) 
  
def nested_check(): 
      
    try: 
        time.sleep(1) 
        nes_nex = chrome.find_element_by_class_name('coreSpriteRightChevron  ') 
        return nes_nex 
      
    except selenium.common.exceptions.NoSuchElementException: 
        return 0

start = time.time()
path()
time.sleep(1) 
url_name('https://www.instagram.com/accounts/login/') 
login(username, password) 
url_name(url) 
download_allposts() 
chrome.close()
end = time.time()
print('[time]', (end-start)/60)
