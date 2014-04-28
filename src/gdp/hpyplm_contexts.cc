#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "hpyplm/hpyplm.h"
#include "corpus/corpus.h"
//do we need all of this?
#include "pyp/m.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

#define kORDER 3  //default 4

using namespace std;
using namespace oxlm;

Dict dict;

/*train and test using given context vectors
 */
int main(int argc, char** argv) {
  if (argc != 4) {
    cerr << argv[0] << " <training.contexts> <test.contexts> <nsamples>\n\nEstimate a " 
         << kORDER << "-gram HPYP LM and report perplexity\n100 is usually sufficient for <nsamples>\n";
    return 1;
  }
  MT19937 eng;
  string train_file = argv[1];
  string test_file = argv[2];
  int samples = atoi(argv[3]);
  
  vector<vector<WordId> > corpuse;
  set<WordId> vocabe, tv;
  const WordId kSOS = dict.Convert("ROOT"); // ("<s>"); //does this conflict with other parts of the code?
  // const WordId kEOS = dict.Convert("</s>");
  cerr << "Reading corpus...\n";
  ReadFromFile(train_file, &dict, &corpuse, &vocabe);
  cerr << "E-corpus size: " << corpuse.size() << " sentences\t (" << vocabe.size() << " word types)\n";
  vector<vector<WordId> > test;
  ReadFromFile(test_file, &dict, &test, &tv);  
  PYPLM<kORDER> lm(vocabe.size(), 1, 1, 1, 1);
  vector<WordId> ctx(kORDER - 1, kSOS);
  for (int sample=0; sample < samples; ++sample) {
    for (const auto& s : corpuse) {
      WordId w = s[0];
      ctx = vector<WordId>(s.begin()+1, s.end());
      if (sample > 0) lm.decrement(w, ctx, eng);
      lm.increment(w, ctx, eng);
      ctx.push_back(w);
    }
    if (sample % 10 == 9) {
      cerr << " [LLH=" << lm.log_likelihood() << "]" << endl;
      if (sample % 30u == 29) lm.resample_hyperparameters(eng);
    } else { cerr << '.' << flush; }
  }
//  lm.print(cerr);
  //unsigned nsent = 1308; //HACK for word prediction (add 1 for end of each test sentence)
  double llh = 0;
  unsigned cnt = 0; // nsent;  //0;
  unsigned oovs = 0;
  for (auto& s : test) {
    WordId w = s[0];
    ctx = vector<WordId>(s.begin()+1, s.end());
    double lp = log(lm.prob(w, ctx)) / log(2);
    if (vocabe.count(w) == 0) {
//      cerr << "**OOV ";
      ++oovs;
      lp = 0;
    }
//      cerr << "p(" << dict.Convert(w) << " |";
//      for (unsigned j = ctx.size() + 1 - kORDER; j < ctx.size(); ++j)
//        cerr << ' ' << dict.Convert(ctx[j]);
//      cerr << ") = " << lp << endl;
    llh -= lp;
    cnt++;
  }
  cnt -= oovs;
  cerr << "  Log_10 prob: " << (-llh * log(2) / log(10)) << endl;
  cerr << "        Count: " << cnt << endl;
  cerr << "         OOVs: " << oovs << endl;
  cerr << "Cross-entropy: " << (llh / cnt) << endl;
  cerr << "   Perplexity: " << pow(2, llh / cnt) << endl;
  return 0;
}

