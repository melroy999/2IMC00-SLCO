package org.xtext.serializer;

import com.google.inject.Inject;
import java.util.List;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.xtext.IGrammarAccess;
import org.eclipse.xtext.RuleCall;
import org.eclipse.xtext.nodemodel.INode;
import org.eclipse.xtext.serializer.analysis.GrammarAlias.AbstractElementAlias;
import org.eclipse.xtext.serializer.analysis.GrammarAlias.TokenAlias;
import org.eclipse.xtext.serializer.analysis.ISyntacticSequencerPDAProvider.ISynNavigable;
import org.eclipse.xtext.serializer.analysis.ISyntacticSequencerPDAProvider.ISynTransition;
import org.eclipse.xtext.serializer.sequencer.AbstractSyntacticSequencer;
import org.xtext.services.TextualSlcoGrammarAccess;

@SuppressWarnings("all")
public class TextualSlcoSyntacticSequencer extends AbstractSyntacticSequencer {

	protected TextualSlcoGrammarAccess grammarAccess;
	protected AbstractElementAlias match_BracketExpression_LeftParenthesisKeyword_0_a;
	protected AbstractElementAlias match_BracketExpression_LeftParenthesisKeyword_0_p;
	protected AbstractElementAlias match_Class_PortsKeyword_3_0_q;
	protected AbstractElementAlias match_Class_StateMachinesKeyword_4_0_q;
	protected AbstractElementAlias match_Class_VariablesKeyword_2_0_q;
	protected AbstractElementAlias match_Model_ChannelsKeyword_5_0_q;
	protected AbstractElementAlias match_Model_ClassesKeyword_3_0_q;
	protected AbstractElementAlias match_Model_ObjectsKeyword_4_0_q;
	protected AbstractElementAlias match_StateMachine_FinalKeyword_5_0_q;
	protected AbstractElementAlias match_StateMachine_InitialKeyword_3_0_q;
	protected AbstractElementAlias match_StateMachine_StateKeyword_4_0_q;
	protected AbstractElementAlias match_StateMachine_TransitionsKeyword_6_0_q;
	protected AbstractElementAlias match_StateMachine_VariablesKeyword_2_0_q;
	
	@Inject
	protected void init(IGrammarAccess access) {
		grammarAccess = (TextualSlcoGrammarAccess) access;
		match_BracketExpression_LeftParenthesisKeyword_0_a = new TokenAlias(true, true, grammarAccess.getBracketExpressionAccess().getLeftParenthesisKeyword_0());
		match_BracketExpression_LeftParenthesisKeyword_0_p = new TokenAlias(true, false, grammarAccess.getBracketExpressionAccess().getLeftParenthesisKeyword_0());
		match_Class_PortsKeyword_3_0_q = new TokenAlias(false, true, grammarAccess.getClassAccess().getPortsKeyword_3_0());
		match_Class_StateMachinesKeyword_4_0_q = new TokenAlias(false, true, grammarAccess.getClassAccess().getStateMachinesKeyword_4_0());
		match_Class_VariablesKeyword_2_0_q = new TokenAlias(false, true, grammarAccess.getClassAccess().getVariablesKeyword_2_0());
		match_Model_ChannelsKeyword_5_0_q = new TokenAlias(false, true, grammarAccess.getModelAccess().getChannelsKeyword_5_0());
		match_Model_ClassesKeyword_3_0_q = new TokenAlias(false, true, grammarAccess.getModelAccess().getClassesKeyword_3_0());
		match_Model_ObjectsKeyword_4_0_q = new TokenAlias(false, true, grammarAccess.getModelAccess().getObjectsKeyword_4_0());
		match_StateMachine_FinalKeyword_5_0_q = new TokenAlias(false, true, grammarAccess.getStateMachineAccess().getFinalKeyword_5_0());
		match_StateMachine_InitialKeyword_3_0_q = new TokenAlias(false, true, grammarAccess.getStateMachineAccess().getInitialKeyword_3_0());
		match_StateMachine_StateKeyword_4_0_q = new TokenAlias(false, true, grammarAccess.getStateMachineAccess().getStateKeyword_4_0());
		match_StateMachine_TransitionsKeyword_6_0_q = new TokenAlias(false, true, grammarAccess.getStateMachineAccess().getTransitionsKeyword_6_0());
		match_StateMachine_VariablesKeyword_2_0_q = new TokenAlias(false, true, grammarAccess.getStateMachineAccess().getVariablesKeyword_2_0());
	}
	
	@Override
	protected String getUnassignedRuleCallToken(EObject semanticObject, RuleCall ruleCall, INode node) {
		return "";
	}
	
	
	@Override
	protected void emitUnassignedTokens(EObject semanticObject, ISynTransition transition, INode fromNode, INode toNode) {
		if (transition.getAmbiguousSyntaxes().isEmpty()) return;
		List<INode> transitionNodes = collectNodes(fromNode, toNode);
		for (AbstractElementAlias syntax : transition.getAmbiguousSyntaxes()) {
			List<INode> syntaxNodes = getNodesFor(transitionNodes, syntax);
			if(match_BracketExpression_LeftParenthesisKeyword_0_a.equals(syntax))
				emit_BracketExpression_LeftParenthesisKeyword_0_a(semanticObject, getLastNavigableState(), syntaxNodes);
			else if(match_BracketExpression_LeftParenthesisKeyword_0_p.equals(syntax))
				emit_BracketExpression_LeftParenthesisKeyword_0_p(semanticObject, getLastNavigableState(), syntaxNodes);
			else if(match_Class_PortsKeyword_3_0_q.equals(syntax))
				emit_Class_PortsKeyword_3_0_q(semanticObject, getLastNavigableState(), syntaxNodes);
			else if(match_Class_StateMachinesKeyword_4_0_q.equals(syntax))
				emit_Class_StateMachinesKeyword_4_0_q(semanticObject, getLastNavigableState(), syntaxNodes);
			else if(match_Class_VariablesKeyword_2_0_q.equals(syntax))
				emit_Class_VariablesKeyword_2_0_q(semanticObject, getLastNavigableState(), syntaxNodes);
			else if(match_Model_ChannelsKeyword_5_0_q.equals(syntax))
				emit_Model_ChannelsKeyword_5_0_q(semanticObject, getLastNavigableState(), syntaxNodes);
			else if(match_Model_ClassesKeyword_3_0_q.equals(syntax))
				emit_Model_ClassesKeyword_3_0_q(semanticObject, getLastNavigableState(), syntaxNodes);
			else if(match_Model_ObjectsKeyword_4_0_q.equals(syntax))
				emit_Model_ObjectsKeyword_4_0_q(semanticObject, getLastNavigableState(), syntaxNodes);
			else if(match_StateMachine_FinalKeyword_5_0_q.equals(syntax))
				emit_StateMachine_FinalKeyword_5_0_q(semanticObject, getLastNavigableState(), syntaxNodes);
			else if(match_StateMachine_InitialKeyword_3_0_q.equals(syntax))
				emit_StateMachine_InitialKeyword_3_0_q(semanticObject, getLastNavigableState(), syntaxNodes);
			else if(match_StateMachine_StateKeyword_4_0_q.equals(syntax))
				emit_StateMachine_StateKeyword_4_0_q(semanticObject, getLastNavigableState(), syntaxNodes);
			else if(match_StateMachine_TransitionsKeyword_6_0_q.equals(syntax))
				emit_StateMachine_TransitionsKeyword_6_0_q(semanticObject, getLastNavigableState(), syntaxNodes);
			else if(match_StateMachine_VariablesKeyword_2_0_q.equals(syntax))
				emit_StateMachine_VariablesKeyword_2_0_q(semanticObject, getLastNavigableState(), syntaxNodes);
			else acceptNodes(getLastNavigableState(), syntaxNodes);
		}
	}

	/**
	 * Syntax:
	 *     '('*
	 */
	protected void emit_BracketExpression_LeftParenthesisKeyword_0_a(EObject semanticObject, ISynNavigable transition, List<INode> nodes) {
		acceptNodes(transition, nodes);
	}
	
	/**
	 * Syntax:
	 *     '('+
	 */
	protected void emit_BracketExpression_LeftParenthesisKeyword_0_p(EObject semanticObject, ISynNavigable transition, List<INode> nodes) {
		acceptNodes(transition, nodes);
	}
	
	/**
	 * Syntax:
	 *     'ports'?
	 */
	protected void emit_Class_PortsKeyword_3_0_q(EObject semanticObject, ISynNavigable transition, List<INode> nodes) {
		acceptNodes(transition, nodes);
	}
	
	/**
	 * Syntax:
	 *     'state machines'?
	 */
	protected void emit_Class_StateMachinesKeyword_4_0_q(EObject semanticObject, ISynNavigable transition, List<INode> nodes) {
		acceptNodes(transition, nodes);
	}
	
	/**
	 * Syntax:
	 *     'variables'?
	 */
	protected void emit_Class_VariablesKeyword_2_0_q(EObject semanticObject, ISynNavigable transition, List<INode> nodes) {
		acceptNodes(transition, nodes);
	}
	
	/**
	 * Syntax:
	 *     'channels'?
	 */
	protected void emit_Model_ChannelsKeyword_5_0_q(EObject semanticObject, ISynNavigable transition, List<INode> nodes) {
		acceptNodes(transition, nodes);
	}
	
	/**
	 * Syntax:
	 *     'classes'?
	 */
	protected void emit_Model_ClassesKeyword_3_0_q(EObject semanticObject, ISynNavigable transition, List<INode> nodes) {
		acceptNodes(transition, nodes);
	}
	
	/**
	 * Syntax:
	 *     'objects'?
	 */
	protected void emit_Model_ObjectsKeyword_4_0_q(EObject semanticObject, ISynNavigable transition, List<INode> nodes) {
		acceptNodes(transition, nodes);
	}
	
	/**
	 * Syntax:
	 *     'final'?
	 */
	protected void emit_StateMachine_FinalKeyword_5_0_q(EObject semanticObject, ISynNavigable transition, List<INode> nodes) {
		acceptNodes(transition, nodes);
	}
	
	/**
	 * Syntax:
	 *     'initial'?
	 */
	protected void emit_StateMachine_InitialKeyword_3_0_q(EObject semanticObject, ISynNavigable transition, List<INode> nodes) {
		acceptNodes(transition, nodes);
	}
	
	/**
	 * Syntax:
	 *     'state'?
	 */
	protected void emit_StateMachine_StateKeyword_4_0_q(EObject semanticObject, ISynNavigable transition, List<INode> nodes) {
		acceptNodes(transition, nodes);
	}
	
	/**
	 * Syntax:
	 *     'transitions'?
	 */
	protected void emit_StateMachine_TransitionsKeyword_6_0_q(EObject semanticObject, ISynNavigable transition, List<INode> nodes) {
		acceptNodes(transition, nodes);
	}
	
	/**
	 * Syntax:
	 *     'variables'?
	 */
	protected void emit_StateMachine_VariablesKeyword_2_0_q(EObject semanticObject, ISynNavigable transition, List<INode> nodes) {
		acceptNodes(transition, nodes);
	}
	
}
