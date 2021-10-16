// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <mkldnn_graph.h>

#include <memory>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNIfNode : public MKLDNNNode {
public:
    MKLDNNIfNode(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void initSupportedPrimitiveDescriptors() override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;

    void inline setExtManager(const MKLDNNExtensionManager::Ptr& extMgr) { ext_mng = extMgr; }

    struct PortMap {
        int from; /**< Index of external/internal out data */
        int to;   /**< Index of external/internal in data */
    };

    class PortMapHelper {
    public:
        PortMapHelper(const MKLDNNMemoryPtr& from, const MKLDNNMemoryPtr& to, const mkldnn::engine& eng);
        virtual ~PortMapHelper() = default;
        virtual void execute(mkldnn::stream& strm);
    protected:
        mkldnn::reorder reorder;
        mkldnn::memory mem_holder_src;
        mkldnn::memory mem_holder_dst;
    };

private:
    MKLDNNExtensionManager::Ptr ext_mng;
    MKLDNNGraph subGraphThen;
    MKLDNNGraph subGraphElse;
    std::deque<MKLDNNMemoryPtr> inputMemThen, inputMemElse, outputMemThen, outputMemElse;

    std::vector<std::shared_ptr<PortMapHelper>>
        beforeThenMappers,
        beforeElseMappers,
        afterThenMappers,
        afterElseMappers;

    std::vector<PortMap>
        thenInputPortMap,
        thenOutputPortMap,
        elseInputPortMap,
        elseOutputPortMap;

    const std::shared_ptr<ov::Node> ovOp;
};

}  // namespace MKLDNNPlugin
