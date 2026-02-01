// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DataLedger {
    mapping(uint256 => bytes32) public txHashes;
    uint256 public txCount;

    event TxStored(uint256 id, bytes32 hash);

    function storeTx(bytes32 txHash) external {
        txHashes[txCount] = txHash;
        emit TxStored(txCount, txHash);
        txCount++;
    }
}
