// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Escrow {
    address public buyer;
    address public seller;
    address public authorizedValidator; // The Consortium Leader/AI Node
    uint256 public amount;
    bool public isReleased;
    bool public isDisputed;

    event FundsDeposited(address buyer, address seller, uint256 amount);
    event FundsReleased(address recipient, uint256 amount, string method);
    event DisputeRaised(address by, string reason);

    constructor(address _buyer, address _seller, address _validator) payable {
        require(msg.value > 0, "ESCROW_ERR: Must deposit funds");
        require(_validator != address(0), "ESCROW_ERR: Invalid validator address");
        
        buyer = _buyer;
        seller = _seller;
        authorizedValidator = _validator;
        amount = msg.value;
        isReleased = false;
        isDisputed = false;
        
        emit FundsDeposited(_buyer, _seller, msg.value);
    }

    /**
     * @dev Release funds. Can be called by the Buyer (manual) 
     * or the Authorized Validator (automatic AI-based release).
     */
    function release() external {
        // Explicit Revert Messages
        require(msg.sender == buyer || msg.sender == authorizedValidator, "ESCROW_ERR: Unauthorized caller");
        require(!isReleased, "ESCROW_ERR: Funds already released");
        require(!isDisputed, "ESCROW_ERR: Cannot release while in dispute");

        isReleased = true;
        
        payable(seller).transfer(amount);
        
        string memory method = (msg.sender == authorizedValidator) ? "AI_VALIDATED" : "BUYER_MANUAL";
        emit FundsReleased(seller, amount, method);
    }

    function raiseDispute(string calldata reason) external {
        require(msg.sender == buyer || msg.sender == seller, "ESCROW_ERR: Only parties can dispute");
        require(!isReleased, "ESCROW_ERR: Funds already released");
        
        isDisputed = true;
        emit DisputeRaised(msg.sender, reason);
    }
}
