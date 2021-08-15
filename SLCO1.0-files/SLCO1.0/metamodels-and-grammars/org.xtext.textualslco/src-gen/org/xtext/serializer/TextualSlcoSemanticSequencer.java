package org.xtext.serializer;

import com.google.inject.Inject;
import com.google.inject.Provider;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.xtext.serializer.acceptor.ISemanticSequenceAcceptor;
import org.eclipse.xtext.serializer.acceptor.SequenceFeeder;
import org.eclipse.xtext.serializer.diagnostic.ISemanticSequencerDiagnosticProvider;
import org.eclipse.xtext.serializer.diagnostic.ISerializationDiagnostic.Acceptor;
import org.eclipse.xtext.serializer.sequencer.AbstractDelegatingSemanticSequencer;
import org.eclipse.xtext.serializer.sequencer.GenericSequencer;
import org.eclipse.xtext.serializer.sequencer.ISemanticNodeProvider.INodesForEObjectProvider;
import org.eclipse.xtext.serializer.sequencer.ISemanticSequencer;
import org.eclipse.xtext.serializer.sequencer.ITransientValueService;
import org.eclipse.xtext.serializer.sequencer.ITransientValueService.ValueTransient;
import org.xtext.services.TextualSlcoGrammarAccess;
import org.xtext.textualSlco.ArgumentType;
import org.xtext.textualSlco.Assignment;
import org.xtext.textualSlco.BidirectionalChannel;
import org.xtext.textualSlco.BinaryOperatorExpression;
import org.xtext.textualSlco.BooleanConstantExpression;
import org.xtext.textualSlco.Delay;
import org.xtext.textualSlco.Final;
import org.xtext.textualSlco.Initial;
import org.xtext.textualSlco.IntegerConstantExpression;
import org.xtext.textualSlco.Model;
import org.xtext.textualSlco.Port;
import org.xtext.textualSlco.PortReference;
import org.xtext.textualSlco.SendSignal;
import org.xtext.textualSlco.SignalArgumentExpression;
import org.xtext.textualSlco.SignalArgumentVariable;
import org.xtext.textualSlco.SignalReception;
import org.xtext.textualSlco.State;
import org.xtext.textualSlco.StateMachine;
import org.xtext.textualSlco.StringConstantExpression;
import org.xtext.textualSlco.TextualSlcoPackage;
import org.xtext.textualSlco.Transition;
import org.xtext.textualSlco.UnidirectionalChannel;
import org.xtext.textualSlco.Variable;
import org.xtext.textualSlco.VariableExpression;
import org.xtext.textualSlco.VariableReference;
import org.xtext.textualSlco.VertexReference;

@SuppressWarnings("all")
public class TextualSlcoSemanticSequencer extends AbstractDelegatingSemanticSequencer {

	@Inject
	private TextualSlcoGrammarAccess grammarAccess;
	
	public void createSequence(EObject context, EObject semanticObject) {
		if(semanticObject.eClass().getEPackage() == TextualSlcoPackage.eINSTANCE) switch(semanticObject.eClass().getClassifierID()) {
			case TextualSlcoPackage.ARGUMENT_TYPE:
				if(context == grammarAccess.getArgumentTypeRule()) {
					sequence_ArgumentType(context, (ArgumentType) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.ASSIGNMENT:
				if(context == grammarAccess.getAssignmentRule() ||
				   context == grammarAccess.getStatementRule()) {
					sequence_Assignment(context, (Assignment) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL:
				if(context == grammarAccess.getBidirectionalChannelRule() ||
				   context == grammarAccess.getChannelRule()) {
					sequence_BidirectionalChannel(context, (BidirectionalChannel) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.BINARY_OPERATOR_EXPRESSION:
				if(context == grammarAccess.getBracketExpressionRule() ||
				   context == grammarAccess.getExpressionRule() ||
				   context == grammarAccess.getExpressionAccess().getBinaryOperatorExpressionOperand1Action_1_0() ||
				   context == grammarAccess.getStatementRule() ||
				   context == grammarAccess.getTerminalExpressionRule()) {
					sequence_Expression(context, (BinaryOperatorExpression) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.BOOLEAN_CONSTANT_EXPRESSION:
				if(context == grammarAccess.getBooleanConstantExpressionRule() ||
				   context == grammarAccess.getBracketExpressionRule() ||
				   context == grammarAccess.getExpressionRule() ||
				   context == grammarAccess.getExpressionAccess().getBinaryOperatorExpressionOperand1Action_1_0() ||
				   context == grammarAccess.getStatementRule() ||
				   context == grammarAccess.getTerminalExpressionRule()) {
					sequence_BooleanConstantExpression(context, (BooleanConstantExpression) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.CLASS:
				if(context == grammarAccess.getClassRule()) {
					sequence_Class(context, (org.xtext.textualSlco.Class) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.DELAY:
				if(context == grammarAccess.getDelayRule() ||
				   context == grammarAccess.getStatementRule()) {
					sequence_Delay(context, (Delay) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.FINAL:
				if(context == grammarAccess.getFinalRule() ||
				   context == grammarAccess.getVertexRule()) {
					sequence_Final(context, (Final) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.INITIAL:
				if(context == grammarAccess.getInitialRule() ||
				   context == grammarAccess.getVertexRule()) {
					sequence_Initial(context, (Initial) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.INTEGER_CONSTANT_EXPRESSION:
				if(context == grammarAccess.getBracketExpressionRule() ||
				   context == grammarAccess.getExpressionRule() ||
				   context == grammarAccess.getExpressionAccess().getBinaryOperatorExpressionOperand1Action_1_0() ||
				   context == grammarAccess.getIntegerConstantExpressionRule() ||
				   context == grammarAccess.getStatementRule() ||
				   context == grammarAccess.getTerminalExpressionRule()) {
					sequence_IntegerConstantExpression(context, (IntegerConstantExpression) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.MODEL:
				if(context == grammarAccess.getModelRule()) {
					sequence_Model(context, (Model) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.OBJECT:
				if(context == grammarAccess.getObjectRule()) {
					sequence_Object(context, (org.xtext.textualSlco.Object) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.PORT:
				if(context == grammarAccess.getPortRule()) {
					sequence_Port(context, (Port) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.PORT_REFERENCE:
				if(context == grammarAccess.getPortReferenceRule()) {
					sequence_PortReference(context, (PortReference) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.SEND_SIGNAL:
				if(context == grammarAccess.getSendSignalRule() ||
				   context == grammarAccess.getStatementRule()) {
					sequence_SendSignal(context, (SendSignal) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.SIGNAL_ARGUMENT_EXPRESSION:
				if(context == grammarAccess.getSignalArgumentRule() ||
				   context == grammarAccess.getSignalArgumentExpressionRule()) {
					sequence_SignalArgumentExpression(context, (SignalArgumentExpression) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.SIGNAL_ARGUMENT_VARIABLE:
				if(context == grammarAccess.getSignalArgumentRule() ||
				   context == grammarAccess.getSignalArgumentVariableRule()) {
					sequence_SignalArgumentVariable(context, (SignalArgumentVariable) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.SIGNAL_RECEPTION:
				if(context == grammarAccess.getSignalReceptionRule() ||
				   context == grammarAccess.getStatementRule()) {
					sequence_SignalReception(context, (SignalReception) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.STATE:
				if(context == grammarAccess.getStateRule() ||
				   context == grammarAccess.getVertexRule()) {
					sequence_State(context, (State) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.STATE_MACHINE:
				if(context == grammarAccess.getStateMachineRule()) {
					sequence_StateMachine(context, (StateMachine) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.STRING_CONSTANT_EXPRESSION:
				if(context == grammarAccess.getBracketExpressionRule() ||
				   context == grammarAccess.getExpressionRule() ||
				   context == grammarAccess.getExpressionAccess().getBinaryOperatorExpressionOperand1Action_1_0() ||
				   context == grammarAccess.getStatementRule() ||
				   context == grammarAccess.getStringConstantExpressionRule() ||
				   context == grammarAccess.getTerminalExpressionRule()) {
					sequence_StringConstantExpression(context, (StringConstantExpression) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.TRANSITION:
				if(context == grammarAccess.getTransitionRule()) {
					sequence_Transition(context, (Transition) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL:
				if(context == grammarAccess.getChannelRule() ||
				   context == grammarAccess.getUnidirectionalChannelRule()) {
					sequence_UnidirectionalChannel(context, (UnidirectionalChannel) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.VARIABLE:
				if(context == grammarAccess.getVariableRule()) {
					sequence_Variable(context, (Variable) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.VARIABLE_EXPRESSION:
				if(context == grammarAccess.getBracketExpressionRule() ||
				   context == grammarAccess.getExpressionRule() ||
				   context == grammarAccess.getExpressionAccess().getBinaryOperatorExpressionOperand1Action_1_0() ||
				   context == grammarAccess.getStatementRule() ||
				   context == grammarAccess.getTerminalExpressionRule() ||
				   context == grammarAccess.getVariableExpressionRule()) {
					sequence_VariableExpression(context, (VariableExpression) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.VARIABLE_REFERENCE:
				if(context == grammarAccess.getVariableReferenceRule()) {
					sequence_VariableReference(context, (VariableReference) semanticObject); 
					return; 
				}
				else break;
			case TextualSlcoPackage.VERTEX_REFERENCE:
				if(context == grammarAccess.getVertexReferenceRule()) {
					sequence_VertexReference(context, (VertexReference) semanticObject); 
					return; 
				}
				else break;
			}
		if (errorAcceptor != null) errorAcceptor.accept(diagnosticProvider.createInvalidContextOrTypeDiagnostic(semanticObject, context));
	}
	
	/**
	 * Constraint:
	 *     type=PrimitiveType
	 */
	protected void sequence_ArgumentType(EObject context, ArgumentType semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.ARGUMENT_TYPE__TYPE) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.ARGUMENT_TYPE__TYPE));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getArgumentTypeAccess().getTypePrimitiveTypeEnumRuleCall_0(), semanticObject.getType());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     (variable=VariableReference expression=Expression)
	 */
	protected void sequence_Assignment(EObject context, Assignment semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.ASSIGNMENT__VARIABLE) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.ASSIGNMENT__VARIABLE));
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.ASSIGNMENT__EXPRESSION) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.ASSIGNMENT__EXPRESSION));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getAssignmentAccess().getVariableVariableReferenceParserRuleCall_0_0(), semanticObject.getVariable());
		feeder.accept(grammarAccess.getAssignmentAccess().getExpressionExpressionParserRuleCall_2_0(), semanticObject.getExpression());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     (
	 *         name=ID 
	 *         (argumentTypes+=ArgumentType argumentTypes+=ArgumentType*)? 
	 *         channelType=ChannelType 
	 *         object1=[Object|ID] 
	 *         port1=PortReference 
	 *         object2=[Object|ID] 
	 *         port2=PortReference
	 *     )
	 */
	protected void sequence_BidirectionalChannel(EObject context, BidirectionalChannel semanticObject) {
		genericSequencer.createSequence(context, semanticObject);
	}
	
	
	/**
	 * Constraint:
	 *     value=BOOLEAN
	 */
	protected void sequence_BooleanConstantExpression(EObject context, BooleanConstantExpression semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.BOOLEAN_CONSTANT_EXPRESSION__VALUE) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.BOOLEAN_CONSTANT_EXPRESSION__VALUE));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getBooleanConstantExpressionAccess().getValueBOOLEANTerminalRuleCall_0(), semanticObject.getValue());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     (name=ID variables+=Variable* ports+=Port* stateMachines+=StateMachine*)
	 */
	protected void sequence_Class(EObject context, org.xtext.textualSlco.Class semanticObject) {
		genericSequencer.createSequence(context, semanticObject);
	}
	
	
	/**
	 * Constraint:
	 *     value=INT
	 */
	protected void sequence_Delay(EObject context, Delay semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.DELAY__VALUE) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.DELAY__VALUE));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getDelayAccess().getValueINTTerminalRuleCall_1_0(), semanticObject.getValue());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     (operand1=Expression_BinaryOperatorExpression_1_0 operator=Operator operand2=Expression)
	 */
	protected void sequence_Expression(EObject context, BinaryOperatorExpression semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.BINARY_OPERATOR_EXPRESSION__OPERAND1) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.BINARY_OPERATOR_EXPRESSION__OPERAND1));
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.BINARY_OPERATOR_EXPRESSION__OPERATOR) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.BINARY_OPERATOR_EXPRESSION__OPERATOR));
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.BINARY_OPERATOR_EXPRESSION__OPERAND2) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.BINARY_OPERATOR_EXPRESSION__OPERAND2));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getExpressionAccess().getBinaryOperatorExpressionOperand1Action_1_0(), semanticObject.getOperand1());
		feeder.accept(grammarAccess.getExpressionAccess().getOperatorOperatorEnumRuleCall_1_1_0(), semanticObject.getOperator());
		feeder.accept(grammarAccess.getExpressionAccess().getOperand2ExpressionParserRuleCall_1_2_0(), semanticObject.getOperand2());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     name=ID
	 */
	protected void sequence_Final(EObject context, Final semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.VERTEX__NAME) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.VERTEX__NAME));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getFinalAccess().getNameIDTerminalRuleCall_0(), semanticObject.getName());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     name=ID
	 */
	protected void sequence_Initial(EObject context, Initial semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.VERTEX__NAME) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.VERTEX__NAME));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getInitialAccess().getNameIDTerminalRuleCall_0(), semanticObject.getName());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     value=INT
	 */
	protected void sequence_IntegerConstantExpression(EObject context, IntegerConstantExpression semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.INTEGER_CONSTANT_EXPRESSION__VALUE) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.INTEGER_CONSTANT_EXPRESSION__VALUE));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getIntegerConstantExpressionAccess().getValueINTTerminalRuleCall_0(), semanticObject.getValue());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     (name=ID classes+=Class* objects+=Object* channels+=Channel*)
	 */
	protected void sequence_Model(EObject context, Model semanticObject) {
		genericSequencer.createSequence(context, semanticObject);
	}
	
	
	/**
	 * Constraint:
	 *     (name=ID class=[Class|ID])
	 */
	protected void sequence_Object(EObject context, org.xtext.textualSlco.Object semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.OBJECT__NAME) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.OBJECT__NAME));
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.OBJECT__CLASS) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.OBJECT__CLASS));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getObjectAccess().getNameIDTerminalRuleCall_0_0(), semanticObject.getName());
		feeder.accept(grammarAccess.getObjectAccess().getClassClassIDTerminalRuleCall_2_0_1(), semanticObject.getClass_());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     name=ID
	 */
	protected void sequence_PortReference(EObject context, PortReference semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.PORT_REFERENCE__NAME) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.PORT_REFERENCE__NAME));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getPortReferenceAccess().getNameIDTerminalRuleCall_0(), semanticObject.getName());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     name=ID
	 */
	protected void sequence_Port(EObject context, Port semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.PORT__NAME) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.PORT__NAME));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getPortAccess().getNameIDTerminalRuleCall_0(), semanticObject.getName());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     (signalName=ID (arguments+=Expression arguments+=Expression*)? port=PortReference)
	 */
	protected void sequence_SendSignal(EObject context, SendSignal semanticObject) {
		genericSequencer.createSequence(context, semanticObject);
	}
	
	
	/**
	 * Constraint:
	 *     expression=Expression
	 */
	protected void sequence_SignalArgumentExpression(EObject context, SignalArgumentExpression semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.SIGNAL_ARGUMENT_EXPRESSION__EXPRESSION) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.SIGNAL_ARGUMENT_EXPRESSION__EXPRESSION));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getSignalArgumentExpressionAccess().getExpressionExpressionParserRuleCall_1_0(), semanticObject.getExpression());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     variable=VariableReference
	 */
	protected void sequence_SignalArgumentVariable(EObject context, SignalArgumentVariable semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.SIGNAL_ARGUMENT_VARIABLE__VARIABLE) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.SIGNAL_ARGUMENT_VARIABLE__VARIABLE));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getSignalArgumentVariableAccess().getVariableVariableReferenceParserRuleCall_0(), semanticObject.getVariable());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     (signalName=ID (arguments+=SignalArgument arguments+=SignalArgument*)? condition=Expression? port=PortReference)
	 */
	protected void sequence_SignalReception(EObject context, SignalReception semanticObject) {
		genericSequencer.createSequence(context, semanticObject);
	}
	
	
	/**
	 * Constraint:
	 *     (
	 *         name=ID 
	 *         variables+=Variable* 
	 *         vertices+=Initial* 
	 *         vertices+=State* 
	 *         vertices+=Final* 
	 *         transitions+=Transition*
	 *     )
	 */
	protected void sequence_StateMachine(EObject context, StateMachine semanticObject) {
		genericSequencer.createSequence(context, semanticObject);
	}
	
	
	/**
	 * Constraint:
	 *     name=ID
	 */
	protected void sequence_State(EObject context, State semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.VERTEX__NAME) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.VERTEX__NAME));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getStateAccess().getNameIDTerminalRuleCall_0(), semanticObject.getName());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     value=STRING
	 */
	protected void sequence_StringConstantExpression(EObject context, StringConstantExpression semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.STRING_CONSTANT_EXPRESSION__VALUE) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.STRING_CONSTANT_EXPRESSION__VALUE));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getStringConstantExpressionAccess().getValueSTRINGTerminalRuleCall_0(), semanticObject.getValue());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     (name=ID source=VertexReference target=VertexReference (statements+=Statement statements+=Statement*)?)
	 */
	protected void sequence_Transition(EObject context, Transition semanticObject) {
		genericSequencer.createSequence(context, semanticObject);
	}
	
	
	/**
	 * Constraint:
	 *     (
	 *         name=ID 
	 *         (argumentTypes+=ArgumentType argumentTypes+=ArgumentType*)? 
	 *         channelType=ChannelType 
	 *         sourceObject=[Object|ID] 
	 *         sourcePort=PortReference 
	 *         targetObject=[Object|ID] 
	 *         targetPort=PortReference
	 *     )
	 */
	protected void sequence_UnidirectionalChannel(EObject context, UnidirectionalChannel semanticObject) {
		genericSequencer.createSequence(context, semanticObject);
	}
	
	
	/**
	 * Constraint:
	 *     variable=VariableReference
	 */
	protected void sequence_VariableExpression(EObject context, VariableExpression semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.VARIABLE_EXPRESSION__VARIABLE) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.VARIABLE_EXPRESSION__VARIABLE));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getVariableExpressionAccess().getVariableVariableReferenceParserRuleCall_0(), semanticObject.getVariable());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     name=ID
	 */
	protected void sequence_VariableReference(EObject context, VariableReference semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.VARIABLE_REFERENCE__NAME) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.VARIABLE_REFERENCE__NAME));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getVariableReferenceAccess().getNameIDTerminalRuleCall_0(), semanticObject.getName());
		feeder.finish();
	}
	
	
	/**
	 * Constraint:
	 *     (type=PrimitiveType name=ID intialValue=Expression?)
	 */
	protected void sequence_Variable(EObject context, Variable semanticObject) {
		genericSequencer.createSequence(context, semanticObject);
	}
	
	
	/**
	 * Constraint:
	 *     name=ID
	 */
	protected void sequence_VertexReference(EObject context, VertexReference semanticObject) {
		if(errorAcceptor != null) {
			if(transientValues.isValueTransient(semanticObject, TextualSlcoPackage.Literals.VERTEX_REFERENCE__NAME) == ValueTransient.YES)
				errorAcceptor.accept(diagnosticProvider.createFeatureValueMissing(semanticObject, TextualSlcoPackage.Literals.VERTEX_REFERENCE__NAME));
		}
		INodesForEObjectProvider nodes = createNodeProvider(semanticObject);
		SequenceFeeder feeder = createSequencerFeeder(semanticObject, nodes);
		feeder.accept(grammarAccess.getVertexReferenceAccess().getNameIDTerminalRuleCall_0(), semanticObject.getName());
		feeder.finish();
	}
}
