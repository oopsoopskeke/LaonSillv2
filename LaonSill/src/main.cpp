#if 1
#include <unistd.h>
#include <fcntl.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>

#include "cuda/Cuda.h"
#include "jsoncpp/json/json.h"

#include "common.h"
#include "DataSet.h"
#include "MockDataSet.h"
#include "Debug.h"
#include "Network.h"
#include "Util.h"
#include "Worker.h"
#include "Job.h"
#include "Communicator.h"
#include "Client.h"
#include "InitParam.h"
#include "Param.h"
#include "ColdLog.h"
#include "SysLog.h"
#include "HotLog.h"
#include "StdOutLog.h"
#include "Perf.h"
#include "Broker.h"
#include "test.h"
#include "ImageUtil.h"
#include "DebugUtil.h"
#include "ResourceManager.h"
#include "PlanOptimizer.h"
#include "WorkContext.h"
#include "PlanParser.h"
#include "ThreadMgmt.h"
#include "Sender.h"
#include "Receiver.h"
#include "Task.h"
#include "MeasureManager.h"
#include "MemoryMgmt.h"

#include "LayerFunc.h"
#include "LayerPropList.h"
#include "Examples.h"

#include "Tools.h"
#include "InputDataProvider.h"
#include "PlanValidator.h"
#include "MeasureLayer.h"
#include "NetworkRecorder.h"

using namespace std;

#ifdef CLIENT_MODE
void printUsageAndExit(char* prog) {
    fprintf(stderr, "Usage: %s [-v] | -t testItemName]\n", prog);
    exit(EXIT_FAILURE);
}

const char          SERVER_HOSTNAME[] = {"localhost"};
int main(int argc, char** argv) {
    int     opt;

    bool    useTestMode = false;

    char*   testItemName;

    // (1) 옵션을 읽는다.
    WorkContext::curBootMode = BootMode::ServerClientMode;
    while ((opt = getopt(argc, argv, "vt:")) != -1) {
        switch (opt) {
        case 'v':
            printf("%s version %d.%d.%d\n", argv[0], SPARAM(VERSION_MAJOR),
                SPARAM(VERSION_MINOR), SPARAM(VERSION_PATCH));
            exit(EXIT_SUCCESS);

        case 't':
            useTestMode = true;
            testItemName = optarg;
            checkTestItem(testItemName);
            break;

        default:    /* ? */
            printUsageAndExit(argv[0]);
            break;
        }
    }

    if (useTestMode) {
        runTest(testItemName);
    } else {
        Client::clientMain(SERVER_HOSTNAME, Communicator::LISTENER_PORT);
    }

    exit(EXIT_SUCCESS);
}

#endif


#ifdef SERVER_MODE
void printUsageAndExit(char* prog) {
    fprintf(stderr,
        "Usage: %s [-v] [-d exampleName | -f networkFilePath | -t testItemName"
        " | -r resumeNetworkID <-p resumeParamName> <-k>]\n", prog);
    exit(EXIT_FAILURE);
}

void developerMain(const char* itemName) {
    STDOUT_LOG("enter developerMain()");

    checkCudaErrors(cudaSetDevice(0));
	checkCUBLAS(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

    Examples::run(itemName);

    STDOUT_LOG("exit developerMain()");
}

void singleJobMain(const char* jobFilePath) {
    STDOUT_LOG("enter single job(%s)", jobFilePath);

    SASSERT(access(jobFilePath, F_OK) != -1,
        "cannot access single job filepath. filepath=%s", jobFilePath);

    checkCudaErrors(cudaSetDevice(0));
	checkCUBLAS(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

    string networkID = PlanParser::loadNetwork(jobFilePath);
    WorkContext::updateNetwork(networkID);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    if ((SNPROP(status) == NetworkStatus::Train) && (SNPROP(measureLayer).size() > 0)) {
        MeasureManager::insertEntry(networkID, SNPROP(measureLayer));
        network->setMeasureInserted();
    }

    if (SNPROP(status) == NetworkStatus::Test) {
        SNPROP(testInterval) = (uint32_t)numeric_limits<uint32_t>::max();
        SNPROP(epochs) = 1;
    }

    PlanOptimizer::buildPlans(networkID);

    if ((NetworkStatus)SNPROP(status) == NetworkStatus::Train) {
        PlanOptimizer::runPlan(networkID, false);
    } else {
        PlanOptimizer::runPlan(networkID, true);
    }

    if (SNPROP(status) == NetworkStatus::Test) {
        for (int i = 0; i < SNPROP(measureLayer).size(); i++) {
            string measureLayerName = SNPROP(measureLayer)[i];
            Layer<float>* layer = network->findLayer(measureLayerName);

            MeasureLayer<float>* measureLayer = 
                dynamic_cast<MeasureLayer<float>*>(layer);
            
            if (measureLayer == NULL)
                continue;

            float measureVal = measureLayer->measureAll();
            if (measureVal != measureVal) // NaN case
                measureVal = 0.0;
            STDOUT_LOG(" - %s : %f\n", measureLayerName.c_str(), measureVal);
        }
    }

    STDOUT_LOG("exit single job(%s)", jobFilePath);
}

void resumeJobMain(const char* networkID, const char* paramName, bool keepHistory) {
    if (paramName == NULL)
        STDOUT_LOG("enter resume job(request network ID=%s)", networkID);
    else
        STDOUT_LOG("enter resume job(request network ID=%s, request param name=%s)", 
                networkID, paramName);

    checkCudaErrors(cudaSetDevice(0));
	checkCUBLAS(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

    string newNetworkID;
    if (paramName == NULL) {
        newNetworkID = Network<float>::createResumeNetwork(string(networkID), "", keepHistory);
    } else {
        newNetworkID = Network<float>::createResumeNetwork(string(networkID), 
                string(paramName), keepHistory);
    }
    WorkContext::updateNetwork(newNetworkID);

    if (SNPROP(status) == NetworkStatus::Test) {
        STDOUT_LOG("resume network needs train network status. exit training(networkID=%s)",
                networkID);
        return;
    }

    PlanOptimizer::buildPlans(newNetworkID);
    PlanOptimizer::runPlan(newNetworkID, false);

    if (paramName == NULL) {
        STDOUT_LOG("exit resume job(request network ID=%s, new network ID=%s)", 
                networkID, newNetworkID.c_str());
    } else {
        STDOUT_LOG(
          "exit resume job(request network ID=%s, request param name=%s, new network ID=%s)", 
            networkID, paramName, newNetworkID.c_str());
    }
}

int main(int argc, char** argv) {
    int     opt;

    // 처음 생각했던 것보다 실행모드의 개수가 늘었다.
    // 모드가 하나만 더 추가되면 그냥 enum type으로 모드를 정의하도록 하자.

    bool    useDeveloperMode = false;
    bool    useSingleJobMode = false;
    bool    useTestMode = false;
    bool    useResumeJobMode = false;

    char*   singleJobFilePath;
    char*   testItemName;
    char*   exampleName;
    char*   resumeNetworkID;
    char*   resumeParamName = NULL;
    bool    keepHistory;

    // (1) 옵션을 읽는다.
    WorkContext::curBootMode = BootMode::ServerClientMode;
    while ((opt = getopt(argc, argv, "vd:f:t:r:p:k")) != -1) {
        switch (opt) {
        case 'v':
            printf("%s version %d.%d.%d\n", argv[0], SPARAM(VERSION_MAJOR),
                SPARAM(VERSION_MINOR), SPARAM(VERSION_PATCH));
            exit(EXIT_SUCCESS);

        case 'd':
            if (useSingleJobMode | useTestMode | useResumeJobMode)
                printUsageAndExit(argv[0]);
            useDeveloperMode = true;
            exampleName = optarg;
            WorkContext::curBootMode = BootMode::DeveloperMode;
            Examples::checkItem(exampleName);
            break;

        case 'f':
            if (useDeveloperMode | useTestMode | useResumeJobMode)
                printUsageAndExit(argv[0]);
            useSingleJobMode = true;
            singleJobFilePath = optarg;
            WorkContext::curBootMode = BootMode::SingleJobMode;
            break;

        case 'r':
            if (useDeveloperMode | useSingleJobMode | useTestMode)
                printUsageAndExit(argv[0]);
            useResumeJobMode = true;
            resumeNetworkID = optarg;
            WorkContext::curBootMode = BootMode::ResumeJobMode;
            break;

        case 'p':
            if (useDeveloperMode | useSingleJobMode | useTestMode)
                printUsageAndExit(argv[0]);
            useResumeJobMode = true;
            resumeParamName = optarg;
            WorkContext::curBootMode = BootMode::ResumeJobMode;
            break;

        case 'k':
            if (useDeveloperMode | useSingleJobMode | useTestMode)
                printUsageAndExit(argv[0]);
            useResumeJobMode = true;
            keepHistory = true;
            WorkContext::curBootMode = BootMode::ResumeJobMode;
            break;

        case 't':
            if (useSingleJobMode | useDeveloperMode | useResumeJobMode)
                printUsageAndExit(argv[0]);
            useTestMode = true;
            testItemName = optarg;
            checkTestItem(testItemName);
            WorkContext::curBootMode = BootMode::TestMode;
            break;

        default:    /* ? */
            printUsageAndExit(argv[0]);
            break;
        }
    }

    // COMMENT: 만약 이후에 인자를 받고 싶다면 optind를 기준으로 인자를 받으면 된다.
    //  ex. Usage: %s [-d | -f jobFilePath] hostPort 와 같은 식이라면
    //  hostPort = atoi(argv[optind]);로 인자값을 받으면 된다.
    //  개인적으로 host port와 같은 정보는 SPARAM으로 정의하는 것을 더 선호한다.

    // (2) 서버 시작 시간 측정을 시작한다.
    struct timespec startTime;
    SPERF_START(SERVER_RUNNING_TIME, &startTime);
	STDOUT_BLOCK(cout << "LAONSILL engine starts" << endl;);

    // (3) 파라미터, 로깅, job 모듈을 초기화 한다.
    MemoryMgmt::init();
    InitParam::init();
    Perf::init();
    SysLog::init();
    ColdLog::init();
    Job::init();
    Task::init();
    Broker::init();
    Network<float>::init();
    MeasureManager::init();

    ResourceManager::init();
    PlanOptimizer::init();
    LayerFunc::init();
    LayerPropList::init();

    int threadCount;    // 쓰레드 개수. flusher thread, main thread 개수는 제외
    if (useDeveloperMode || useSingleJobMode || useResumeJobMode) {
        threadCount = 0;
    } else {
        threadCount = 1;
    }

    if (threadCount > 0) {
        // 핫 로그는 여러쓰레드에서 동작하게 되어 있다.
        threadCount = ThreadMgmt::init();
        HotLog::init();
    	HotLog::launchThread(threadCount);

        InputDataProvider::init();
    }
    SYS_LOG("Logging system is initialized...");

    // (4) 뉴럴 네트워크 관련 기본 설정을 한다.
	cout.precision(SPARAM(COUT_PRECISION));
	cout.setf(ios::fixed);
	Util::setOutstream(&cout);
	Util::setPrint(false);

    // (5) 모드에 따른 동작을 수행한다.
    // DeveloperMode와 SingleJobMode는 1쓰레드(메인쓰레드)로만 동작을 한다.
    // TestMode와 DefaultMode(ServerClientMode)는 여러 쓰레드로 동작을 하게 된다.
    if (useDeveloperMode) {
        // (5-A-1) Cuda를 생성한다.
        Cuda::create(SPARAM(GPU_COUNT));
        COLD_LOG(ColdLog::INFO, true, "CUDA is initialized");

        // (5-A-2) DeveloperMain()함수를 호출한다.
        developerMain(exampleName);

        // (5-A-3) 자원을 해제 한다.
        Cuda::destroy();
    } else if (useSingleJobMode) {
        // (5-B-1) Cuda를 생성한다.
        Cuda::create(SPARAM(GPU_COUNT));
        COLD_LOG(ColdLog::INFO, true, "CUDA is initialized");

        // (5-B-2) DeveloperMain()함수를 호출한다.
        singleJobMain(singleJobFilePath);

        // (5-B-3) 자원을 해제 한다.
        Cuda::destroy();
    } else if (useResumeJobMode) {
        // (5-B-1) Cuda를 생성한다.
        Cuda::create(SPARAM(GPU_COUNT));
        COLD_LOG(ColdLog::INFO, true, "CUDA is initialized");

        // (5-B-2) DeveloperMain()함수를 호출한다.
        resumeJobMain(resumeNetworkID, resumeParamName, keepHistory);

        // (5-B-3) 자원을 해제 한다.
        Cuda::destroy();
    } else if (useTestMode) {
        // (5-D-1) Producer&Consumer를 생성.
        Worker::launchThreads(SPARAM(GPU_COUNT), SPARAM(JOB_CONSUMER_COUNT));
        Sender::launchThread();
        Receiver::launchThread();

        // (5-D-2) Listener & Sess threads를 생성.
        Communicator::launchThreads(SPARAM(SESS_COUNT));

        // (5-D-3) developer mode 처럼 테스트하는 케이스도 있어서...
        checkCudaErrors(cudaSetDevice(0));
        checkCUBLAS(cublasCreate(&Cuda::cublasHandle));
        checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

        while (!ThreadMgmt::isReady()) {
            sleep(0);
        }

        // (5-D-4) 테스트를 실행한다.
        runTest(testItemName);

        // (5-D-5) release resources
        Job* haltJob = new Job(JobType::HaltMachine);
        Worker::pushJob(haltJob);

        Communicator::halt();       // threads will be eventually halt

        // (5-D-6) 각각의 쓰레드들의 종료를 기다린다.
        Worker::joinThreads();
        Communicator::joinThreads();

        // (5-D-7) 자원을 해제 한다.
        Cuda::destroy();
    } else {
        // (5-E-1) Producer&Consumer를 생성.
        Worker::launchThreads(SPARAM(GPU_COUNT), SPARAM(JOB_CONSUMER_COUNT));
        Sender::launchThread();
        Receiver::launchThread();

        // (5-E-2) Listener & Sess threads를 생성.
        Communicator::launchThreads(SPARAM(SESS_COUNT));

        // (5-E-3) 각각의 쓰레드들의 종료를 기다린다.
        Worker::joinThreads();
        Communicator::joinThreads();
    }

    Task::destroy();
    LayerFunc::destroy();
    // (6) 로깅 관련 모듈이 점유했던 자원을 해제한다.
    if (threadCount > 0) {
        ThreadMgmt::destroy();
        HotLog::destroy();
    }
    ColdLog::destroy();
    SysLog::destroy();
    Broker::destroy();

    // (7) 서버 종료 시간을 측정하고, 계산하여 서버 실행 시간을 출력한다.
    SPERF_END(SERVER_RUNNING_TIME, startTime);
    STDOUT_LOG("server running time : %lf\n", SPERF_TIME(SERVER_RUNNING_TIME));
	STDOUT_BLOCK(cout << "LAONSILL engine ends" << endl;);

    InitParam::destroy();
	exit(EXIT_SUCCESS);
}
#endif

#endif
