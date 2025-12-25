#include "acl/acl.h"
#include "cstdint"

enum ZCCLDataType {
    ZCCL_DATA_TYPE_INT8 = 0;
    ZCCL_DATA_TYPE_INT16 = 1;
    ZCCL_DATA_TYPE_INT32 = 2;
    ZCCL_DATA_TYPE_FP16 = 3;
    ZCCL_DATA_TYPE_FP32 = 4;
    ZCCL_DATA_TYPE_INT64 = 5;
    ZCCL_DATA_TYPE_BFP16 = 6;
};

namespace sglang {
namespace zccl {

size_t getSizeFromTypeEnum(ZCCLDataType dtype)
{
    switch (dtype) {
        case ZCCLDataType::ZCCL_DATA_TYPE_INT8:
            return sizeof(int8_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT16:
            return sizeof(int16_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT32:
            return sizeof(int32_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_INT64:
            return sizeof(int64_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_FP16:
            return sizeof(int16_t);
        case ZCCLDataType::ZCCL_DATA_TYPE_FP32:
            return sizeof(float);
        case ZCCLDataType::ZCCL_DATA_TYPE_BFP16:
            return sizeof(int16_t);
        default:
            break;
    }
}

extern "C" int zccl_all_gather(void *input, void *output, uint64_t numel, ZCCLDataType data_type, int team_id, aclrtStream stream);

extern "C" void zcclReduceScatter(uint8_t *inp, uint8_t *out,
    size_t inpNumel, ZCCLDataType dataType, int teamId, aclrtStream stream, uint32_t reduceOp = 0);

}  // namespace zccl
}  // namespace sglang
