import subprocess
import os

def GetTOPMedResults(CMD, PWD, DIR, CH=['chr' + str(x) for x in range(1, 23)]):

    if not os.path.exists(DIR):
        os.mkdir(DIR)

    os.chdir(DIR)

    subprocess.call(CMD, shell=True)

    Fs = [x for x in os.listdir('.') if x.find('chr') == 0 and x.endswith('.zip')]

    for F in Fs:
        subprocess.call('unzip -P %s %s'%(PWD, F), shell=True)
    
    for ch in CH:
        subprocess.call('tabix -p vcf %s.dose.vcf.gz'%ch, shell=True)
    

    Fs2 = [x + '.dose.vcf.gz' for x in CH]
    ouF = DIR + 'imputated'
    subprocess.call('bcftools concat %s -O z -o %s.vcf.gz'%(' '.join(Fs2), ouF), shell=True)
    subprocess.call('tabix -p vcf %s.vcf.gz'%ouF, shell=True)



CMD = 'curl -sL https://imputation.biodatacatalyst.nhlbi.nih.gov/get/1335972/63b34c1040317038847773b9f2a52d1dca2e4d7a222e3be39d7eedf920928dc9 | bash'
PWD = '..'
DIR='GenotypingStanford896S'
GetTOPMedResults(CMD, PWD, DIR)
