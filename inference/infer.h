/*!
 * \file infer.h
 * \brief data infer
 */
#ifndef LIGHTLDA_INFER_H_
#define LIGHTLDA_INFER_H_

#include "common.h"
#include "alias_table.h"
#include "data_stream.h"
#include "data_block.h"
#include "document.h"
#include "meta.h"
#include "util.h"
#include "model.h"
#include "inferer.h"
#include <vector>
#include <iostream>
#include <thread>
// #include <pthread.h>
#include <multiverso/barrier.h>
#include <multiverso/stop_watch.h>
#include "dump.h"

namespace multiverso { namespace lightlda
{     
    class Infer
    {
    public:
        static void Init();
        static void Clear();
        static std::vector<std::pair<int32_t, int32_t>> predict(std::vector<std::string> &tokens_input);
    private:
        static void Inference(std::vector<Inferer*>& inferers);
        static void* InferenceThread(void* arg);
        static void InitDocument();
        static std::vector<std::pair<int32_t,int32_t>> DumpDocTopic();
    private:
        /*! \brief training data access */
        static IDataStream* data_stream;
        /*! \brief training data meta information */
        static Meta meta;
        static dump *dmp;
        static LocalModel* model;
    };
    
} // namespace lightlda
} // namespace multiverso

#endif // LIGHTLDA_INFER_H_

