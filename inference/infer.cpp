#include "infer.h"

namespace multiverso { namespace lightlda
{
    IDataStream* Infer::data_stream = nullptr;
    Meta Infer::meta;
    dump* Infer::dmp = nullptr;
    LocalModel* Infer::model = nullptr;     

    void Infer::Init()
    {
        multiverso::lightlda::Config::inference = true;
        dmp = new dump(Config::input_dir);
        model = new LocalModel(); 
    }

    void Infer::Clear()
    {
        delete model;
        delete dmp;
    }

    std::vector<std::pair<int32_t, int32_t>> Infer::\
            predict(std::vector<std::string> &tokens_input)
    {
        dmp -> binary_dump(tokens_input);
        meta.Init(dmp);
        //init model using dump
		    model->Init(dmp, &meta);
        //init document stream
        data_stream = CreateDataStream(dmp);
        //init documents
        InitDocument();
        //init alias table
        AliasTable* alias_table = new AliasTable();
        //init inferers
        std::vector<Inferer*> inferers;
        Barrier barrier(Config::num_local_workers);

        for (int32_t i = 0; i < Config::num_local_workers; ++i)
        {
            inferers.push_back(new Inferer(alias_table, data_stream, 
                &meta, model, 
                &barrier, i, Config::num_local_workers));
        }

        Inference(inferers);
        //elapsedSeconds += watch.ElapsedSeconds();
        //dump doc topic
        std::vector<std::pair<int32_t, int32_t>> tokens = DumpDocTopic();
        //watch.Restart();
   
    //recycle space
        for (auto& inferer : inferers)
        {
            delete inferer;
            inferer = nullptr;
        }

    // pthread_barrier_destroy(&barrier);
        delete data_stream;
        delete alias_table;
        model->ClearTable();
        meta.Clear();
        return tokens;
    }

    void Infer::Inference(std::vector<Inferer*>& inferers)
    {
        //pthread_t * threads = new pthread_t[Config::num_local_workers];
        //if(nullptr == threads)
        //{
        //    Log::Fatal("failed to allocate space for worker threads");
        //}
        std::vector<std::thread> threads;
        for(int32_t i = 0; i < Config::num_local_workers; ++i)
        {
            threads.push_back(std::thread(&InferenceThread, inferers[i]));
            //if(pthread_create(threads + i, nullptr, InferenceThread, inferers[i]))
            //{
            //    Log::Fatal("failed to create worker threads");
            //}
        }
        for(int32_t i = 0; i < Config::num_local_workers; ++i)
        {
            // pthread_join(threads[i], nullptr);
            threads[i].join();
        }
        // delete [] threads;
    }

    void* Infer::InferenceThread(void* arg)
    {
        Inferer* inferer = (Inferer*)arg;
        // inference corpus block by block
        for (int32_t block = 0; block < Config::num_blocks; ++block)
        {
            inferer->BeforeIteration(block);
            for (int32_t i = 0; i < Config::num_iterations; ++i)
            {
                inferer->DoIteration(i);
            }
            inferer->EndIteration();
        }
        return nullptr;
    }

    void Infer::InitDocument()
    {
        xorshift_rng rng;
        for (int32_t block = 0; block < Config::num_blocks; ++block)
        {
            data_stream->BeforeDataAccess();
            DataBlock& data_block = data_stream->CurrDataBlock();
            int32_t num_slice = meta.local_vocab(block).num_slice();
            for (int32_t slice = 0; slice < num_slice; ++slice)
            {
                for (int32_t i = 0; i < data_block.Size(); ++i)
                {
                    Document* doc = data_block.GetOneDoc(i);
                    int32_t& cursor = doc->Cursor();
                    if (slice == 0) cursor = 0;
                    int32_t last_word = meta.local_vocab(block).LastWord(slice);
                    for (; cursor < doc->Size(); ++cursor)
                    {
                        if (doc->Word(cursor) > last_word) break;
                        // Init the latent variable
                        if (!Config::warm_start)
                            doc->SetTopic(cursor, rng.rand_k(Config::num_topics));
                    }
                }
            }
            data_stream->EndDataAccess();
        }
    }

    std::vector<std::pair<int32_t,int32_t>> \
            Infer::DumpDocTopic()
    {
        std::vector<std::pair<int32_t,int32_t>> tokens;
        Row<int32_t> doc_topic_counter(0, Format::Sparse, kMaxDocLength); 
        for (int32_t block = 0; block < Config::num_blocks; ++block)
        {
            data_stream->BeforeDataAccess();
            DataBlock& data_block = data_stream->CurrDataBlock();
            for (int i = 0; i < data_block.Size(); ++i)
            {
                Document* doc = data_block.GetOneDoc(i);
                doc_topic_counter.Clear();
                doc->GetDocTopicVector(doc_topic_counter);
                Row<int32_t>::iterator iter = doc_topic_counter.Iterator();
                while (iter.HasNext())
                {
                    tokens.push_back(std::make_pair(iter.Key(), iter.Value()));
                    iter.Next();
                }
            }
            data_stream->EndDataAccess();
        }
        return tokens;
    }
} // namespace lightlda
} // namespace multiverso

