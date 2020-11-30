#!/usr/bin/python
#
# FIXME: should check error but .... it is a very trivial program & I'm too lazy :)
#
import os
from porter2stemmer import Porter2Stemmer

def doStem(word):
    stemmer = Porter2Stemmer()
    return stemmer.stem(word)

tarDirPath = "/data/etri/elsevier"
tmpDirPath = "/data/etri/tmp"
targetDirPath = "/data/etri/flatten"

globalKeywords = []
globalRawKeywords = []

def genDir():
    if 'cnt' not in genDir.__dict__:
        genDir.cnt = 0
    dirPath = targetDirPath + "/%.12d" % genDir.cnt
    genDir.cnt = genDir.cnt + 1
    os.mkdir(dirPath)
    return dirPath

def checkCeKeyword(idx, content):
    if content[idx:idx+10] == 'ce:keyword':
        return True
    return False

def checkCeText(idx, content):
    if content[idx:idx+8] == 'ce:text>':
        return True
    return False

def getKeyword(idx, content):
    result = ""

    for i in range(100):
        if content[idx + i] == '<':
            break
        result = result + content[idx + i]

    return result

def getKeywords(filePath):
    global globalKeywords
    global globalRawKeywords
    file = open(filePath, 'r')
    content = file.read()

    step = 0        # find ce:keyword

    keywords = []

    step = 0
    contentLen = len(content)

    # very inefficient.. but i don't care.. 
    for idx in range(contentLen - 20):
        if step == 0:       # finding ce:keyword
            if content[idx] == 'c':
                if checkCeKeyword(idx, content) == True:
                    step = 1
        elif step == 1:       # finding ce:text
            if content[idx] == 'c':
                if checkCeText(idx, content) == True:
                    keyword = getKeyword(idx + 8, content)
                    stemmedKeyword = doStem(keyword)
                    keywords.append(stemmedKeyword)

                    if stemmedKeyword != "":
                        globalKeywords.append(stemmedKeyword)
                        globalRawKeywords.append(keyword)
                    step = 2
        elif step == 2:     # finding <ce:text 
            if content[idx] == 'c':
                if checkCeKeyword(idx, content) == True:
                    step = 0

    file.close()

    return keywords

def copyJPEGFiles(source, target):
    for fileInfo in os.walk(source):
        jpgFiles = fileInfo[2]
    
        if len(fileInfo[1]) > 0:
            continue

        for jpgFile in jpgFiles:
            if len(jpgFile) < 4:
                continue
            if jpgFile[-4:] == ".jpg":
                os.system("cp %s/%s %s/." % (fileInfo[0], jpgFile, target))

def copyEssentialFiles(dirPath, depth):
    if depth == 4:
        mainXMLFilePath = "%s/main.xml" % dirPath
        if not os.path.isfile(mainXMLFilePath):
            return

        keywords = getKeywords(mainXMLFilePath)
        if len(keywords) == 0:
            return

        newDirPath = genDir()
        os.system("cp %s %s/." % (mainXMLFilePath, newDirPath))
        copyJPEGFiles(dirPath, newDirPath) 

        file = open('%s/keywords.txt' % newDirPath, 'w')
        for keyword in keywords:
            file.write(keyword + '\n')
        file.close()

    else:
        for filename in os.listdir(dirPath):
            if filename == "..":
                continue
            if filename == ".":
                continue

            filePath = "%s/%s" % (dirPath, filename)

            if os.path.isfile(filePath):
                continue

            copyEssentialFiles(filePath, depth + 1)

def preprocess():
    global globalKeywords
    global globalRawKeywords

    fileKeyword = open('/data/etri/flatten/keywords.txt', 'w')
    fileRawKeyword = open('/data/etri/flatten/rawkeywords.txt', 'w')

    targetDirList = os.listdir(tarDirPath)
    step = 1
    for filename in targetDirList:
        if filename.endswith(".tar"):
            print 'processing %d/%d' % (step, len(targetDirList))
            os.system("tar xf %s/%s -C %s" % (tarDirPath, filename, tmpDirPath))
            copyEssentialFiles("%s" % tmpDirPath, 0)
            os.system("rm -rf %s/*" % tmpDirPath)

            step = step + 1

            for keyword in globalKeywords:
                fileKeyword.write(keyword + '\n')
            globalKeywords = []
            fileKeyword.flush()

            for keyword in globalRawKeywords:
                fileRawKeyword.write(keyword + '\n')
            globalRawKeywords = []
            fileRawKeyword.flush()

    fileKeyword.close()
    fileRawKeyword.close()

def getTop1000Keywords():
    wordDict = dict()
    duplicateCount = 0
    index = 0

    file = open('/data/etri/flatten/keywords.txt', 'r')

    while True:
        line = file.readline()
        if line == '':
            break

        word = line[0:-1]
            
        if word not in wordDict:
            wordDict[word] = 1
        else:
            wordDict[word] = wordDict[word] + 1
            duplicateCount = duplicateCount + 1

        index = index + 1

        if index % 10000 == 0:
            print "process %d line" % index

    file.close()

    maxWordCount = 0
    wordCountDict = dict()

    for i in range(10000):
        wordCountDict[i] = []

    for duplicatedWord in wordDict:
        count = wordDict[duplicatedWord]
        if maxWordCount < count:
            maxWordCount = count

        if count > 10000:
            print 'duplicate word count is more than 10000. count=%d' % count
            exit(0)

        wordCountDict[count].append(duplicatedWord)


    index = maxWordCount
    remainSlot = 1000
    top1000Slot = []

    print 'max word count : %d' % index

    while True:
        curCount = len(wordCountDict[index])
        if remainSlot < curCount:
            print 'index : %d, remainsSlot : %d, curCount : %d' %\
                (index, remainSlot, curCount)
            break

        for word in wordCountDict[index]:
            top1000Slot.append(word)

        remainSlot = remainSlot - curCount
        index = index - 1

    newfile = open('/data/etri/flatten/top1000keywords.txt', 'w')
    for duplicatedWord in top1000Slot:
        newfile.write('%s\n' % duplicatedWord)

    newfile.close()

preprocess()
getTop1000Keywords()
