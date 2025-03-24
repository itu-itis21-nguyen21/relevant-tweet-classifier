import requests

# Define the API URL
url = 'http://127.0.0.1:8000/predict'

# Sample data
data = {
    "texts" : ["Biz hep ne diyorduk, İbrahim Erdemoğlu dedikleri adamdan bir şey olmaz #sasa",
                "Sasa ve Hektaş’ın aynı holdinge bağlı olduğuna dair saçma bir iddia mı var? Ne alaka ya.",
                "Yeni bir ürün geliştirdik ve temizlik konusunda oldukça başarılı olduğunu belirledik #SASA #Borsa #temizlik",
                "Endekse rağmen ayakta duran, bir dönem yatırımcıların favorisi olup şimdi nefret edilen ve en önemlisi endeks etkisi yüksek olan bir firma gözüme çarptı… #SASA Teknik analizini ‘Haftalık Hisse Analizi’ videomda yaptım… İzlemek için 👉 https://t.co/ocVFEERXBR",
                "YEDIBAHIS 250 TL Yatirim Sartsiz Deneme Bonusu! TEK 1 YATIRIMA 3 BONUS FIRSATI! 2.000.000 TL SLOT TURNUVASI! GUNLUK 1.000.000 TL ÇEKIM! #sasa",
                "%100 Burs imkanı için 6 Ocak bursluluk sınavımıza bekliyoruz. BİLTES KOLEJİ",
                "Tarafınıza 11Asliye Ceza Mahkemesi Tarafınca Hukuk Süreci Başlatılmıştır.Uygulama üzerinden dava detaylarına erişebilirsiniz",
                "SEKER GIBI BIR GUN ICIN DOKTOR TAVSIYESI ! Sweet Bonanza'da Gecerli 100 Freespin Promokod: SEKERLIGUN500X BANKA GIBI SITE",
                "@i_Erdemoglu ibrahim bey size ve babanızın size bırakmış olduğu öhütlere güvenerek. Hayal kırıklığına uğrattiniz bizlere. Hisse gideceği zaman çıkıp açıklama yapıyorsunuz. Yapmayın ah alıyorsunuz. Dolar olarak 2021 yılında şuan. son 3 sene içinde kim giriş yaptıysa zararda.",
                "Servet Vergisi Alınsın” diyen SASA'nın patronu İbrahim Erdemoğlu’na Mahfi Eğilmez’den destek.. https://t.co/CvRXuQ6aXB",
                "İbrahim Erdemoğlu: Servet Vergisi gelsin https://t.co/U5s8Lg81yf @istegundemcom aracılığıyla",
                "@ZAMajans Bizim örf ve adetlerimiz genellikle damadı sikmeye yönelik. Bazı kadınlarımız var ki yok illa karaca olsun dyson olsun beyaz eşya bosh olsun halı merinos olsun. Daha makul ve aynı işlevi gören başka marka olsa olmuyor mu. Sen ne vadediyorsun peki permatikle traşlanmış am dışında?",
                "@furkancerkes Merinos furki,bu adam kırmızı halı sererek,paçavrası Türk bayrağının yanında göndere çekilerek karşılanırken zoruna gitmedide,bir gazeteci böyle yazınca mı zoruna gitti? H S ordan!!!",
                "@Selin_seki Merinos hali gibi göğsü",
                "Saltanat kalkacak,Halifelik bitti,Kadın Erkek eşit,Şapkasız çıkmam abi,Tekke zaviye hikaye,SOY adın olacak,– Dur Ata’m yapma!!Derken biz Geometri kitabı yazmış ayyaş ama daha bitmedi.Türk dili,Türk harfi,Türk tarih kurumu,Ankara Hukuk,Merinos Halı,"
            ]
}

# Make the POST request
response = requests.post(url, json=data)
# Print the response
print(response.json())
