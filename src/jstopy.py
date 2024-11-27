from urllib.parse import quote
from PIL import Image
import time
import datetime
import random
import aiohttp
import asyncio
import requests
import pytesseract

now_time = time.localtime()
local_times_format = time.strftime("%Y-%m-%d+%H:%M:%S",now_time).replace(":","%3A")

def outTradeNo():
    now = datetime.datetime.now()
    # 格式化日期时间部分
    date_time_str = now.strftime('%Y%m%d%H%M%S')
    # 生成8位随机数
    random_numbers = "".join(str(random.randint(0, 9)) for _ in range(8))
    out_trade_no = date_time_str + random_numbers
    return out_trade_no

proxy_url = "http://218.203.164.245:9980/style/school/pc/proxy.jsp?url="
params = "optype=apply&UserName=18823768056&CodeType=0&LocalTime={0}&UUID={1}".format(local_times_format,outTradeNo())
verifycode_url = "http://127.0.0.1:9880/verifycode"
final_params_encode = quote(verifycode_url + "?" + params,safe="")
final_url = proxy_url + final_params_encode

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        await fetch(session, final_url)

#短信发送

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()

"""
paramStr（未验证）paramStr 是用户的认证参数 | 
将其添加到请求头中再请求生成图像验证码的URL 理论上这个时候我的网络认证图像验证码
就是返回的图像中的
"""
paramStr = """h1fVAB6tzagRUTAod6ZSXgawizrBhi8KEqi21j7/ffYVgIDC6pxxov+en87zXyYjeAFSSu8IeBeoDpj1hL1DpArk9nlVDgTlFqnSI4IIO7iDhSzzQ9hXEkdMkB1Qp7OlLBqsqJ8LFI7Dtvdykoyh4qytYJgFvHHo/eSCgqLmTuEGsIs6wYYvCowwLfvKIDVd9r9u0sX2gTq7KQXAsprGIovSRCDKXWjSDUOAqdjvqzS41gu+NXbmkE9ZJX7hhDHg+vodJmrNyPoYOo7G1BXpEg=="""
paramStr = input("paramStr：")
verify_code_create_url = "http://218.203.164.245:9980/createVerifycode"
verify_code_referer = """http://218.203.164.245:9980/style/school/pc/index.jsp?paramStr={}""".format(paramStr)
verify_code_headers = {
    "Referer":verify_code_referer
}

path = "./data/test/verify.jpg"
with open(path,"wb") as verifyImg:
    verifyImg.write(requests.get(verify_code_create_url,headers=verify_code_headers).content)
    verifyImg.close()

verify_code = pytesseract.image_to_string(Image.open(path))[:4]
verify_code = str(input("图形验证："))
print("verify_code",verify_code)
message_code = str(input("验证码：")) 

auth_params = """UserType=2&paramStr={0}&province=&pwdType=2&serviceType=301&aidcauthtype=1&vfcodeflg=true&UserName=18418613417&PassWord={1}&verifycode={2}""".format(paramStr,message_code,verify_code)
auth_url = "http://218.203.164.245:9980/page_auth.jsp"
response = requests.post(url=auth_url + "?" + auth_params,headers = verify_code_headers)
print(response.text)


