#ifndef _PYP_CONSTANTS_H_
#define _PYP_CONSTANTS_H_

namespace oxlm {

// Constants that define the order of PYP models. This has to be fixed at
// compile-time due to the template implementation of CRPs.
#define wordLMOrderAS 6
#define wordTagLMOrderAS 6
#define tagLMOrderAS 9
#define actionLMOrderAS 9
#define wordLMOrder 5
#define charLMOrder 10

}  // namespace oxlm

#endif
