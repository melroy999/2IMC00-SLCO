/**
 */
package org.xtext.textualSlco.util;

import org.eclipse.emf.common.notify.Adapter;
import org.eclipse.emf.common.notify.Notifier;

import org.eclipse.emf.common.notify.impl.AdapterFactoryImpl;

import org.eclipse.emf.ecore.EObject;

import org.xtext.textualSlco.ArgumentType;
import org.xtext.textualSlco.Assignment;
import org.xtext.textualSlco.BidirectionalChannel;
import org.xtext.textualSlco.BinaryOperatorExpression;
import org.xtext.textualSlco.BooleanConstantExpression;
import org.xtext.textualSlco.Channel;
import org.xtext.textualSlco.Delay;
import org.xtext.textualSlco.Expression;
import org.xtext.textualSlco.Final;
import org.xtext.textualSlco.Initial;
import org.xtext.textualSlco.IntegerConstantExpression;
import org.xtext.textualSlco.Model;
import org.xtext.textualSlco.Port;
import org.xtext.textualSlco.PortReference;
import org.xtext.textualSlco.SendSignal;
import org.xtext.textualSlco.SignalArgument;
import org.xtext.textualSlco.SignalArgumentExpression;
import org.xtext.textualSlco.SignalArgumentVariable;
import org.xtext.textualSlco.SignalReception;
import org.xtext.textualSlco.State;
import org.xtext.textualSlco.StateMachine;
import org.xtext.textualSlco.Statement;
import org.xtext.textualSlco.StringConstantExpression;
import org.xtext.textualSlco.TextualSlcoPackage;
import org.xtext.textualSlco.Transition;
import org.xtext.textualSlco.UnidirectionalChannel;
import org.xtext.textualSlco.Variable;
import org.xtext.textualSlco.VariableExpression;
import org.xtext.textualSlco.VariableReference;
import org.xtext.textualSlco.Vertex;
import org.xtext.textualSlco.VertexReference;

/**
 * <!-- begin-user-doc -->
 * The <b>Adapter Factory</b> for the model.
 * It provides an adapter <code>createXXX</code> method for each class of the model.
 * <!-- end-user-doc -->
 * @see org.xtext.textualSlco.TextualSlcoPackage
 * @generated
 */
public class TextualSlcoAdapterFactory extends AdapterFactoryImpl
{
  /**
   * The cached model package.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected static TextualSlcoPackage modelPackage;

  /**
   * Creates an instance of the adapter factory.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public TextualSlcoAdapterFactory()
  {
    if (modelPackage == null)
    {
      modelPackage = TextualSlcoPackage.eINSTANCE;
    }
  }

  /**
   * Returns whether this factory is applicable for the type of the object.
   * <!-- begin-user-doc -->
   * This implementation returns <code>true</code> if the object is either the model's package or is an instance object of the model.
   * <!-- end-user-doc -->
   * @return whether this factory is applicable for the type of the object.
   * @generated
   */
  @Override
  public boolean isFactoryForType(Object object)
  {
    if (object == modelPackage)
    {
      return true;
    }
    if (object instanceof EObject)
    {
      return ((EObject)object).eClass().getEPackage() == modelPackage;
    }
    return false;
  }

  /**
   * The switch that delegates to the <code>createXXX</code> methods.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected TextualSlcoSwitch<Adapter> modelSwitch =
    new TextualSlcoSwitch<Adapter>()
    {
      @Override
      public Adapter caseModel(Model object)
      {
        return createModelAdapter();
      }
      @Override
      public Adapter caseClass(org.xtext.textualSlco.Class object)
      {
        return createClassAdapter();
      }
      @Override
      public Adapter caseStateMachine(StateMachine object)
      {
        return createStateMachineAdapter();
      }
      @Override
      public Adapter caseVertex(Vertex object)
      {
        return createVertexAdapter();
      }
      @Override
      public Adapter caseInitial(Initial object)
      {
        return createInitialAdapter();
      }
      @Override
      public Adapter caseState(State object)
      {
        return createStateAdapter();
      }
      @Override
      public Adapter caseFinal(Final object)
      {
        return createFinalAdapter();
      }
      @Override
      public Adapter caseVertexReference(VertexReference object)
      {
        return createVertexReferenceAdapter();
      }
      @Override
      public Adapter caseTransition(Transition object)
      {
        return createTransitionAdapter();
      }
      @Override
      public Adapter caseStatement(Statement object)
      {
        return createStatementAdapter();
      }
      @Override
      public Adapter caseDelay(Delay object)
      {
        return createDelayAdapter();
      }
      @Override
      public Adapter casePortReference(PortReference object)
      {
        return createPortReferenceAdapter();
      }
      @Override
      public Adapter caseSignalReception(SignalReception object)
      {
        return createSignalReceptionAdapter();
      }
      @Override
      public Adapter caseSignalArgument(SignalArgument object)
      {
        return createSignalArgumentAdapter();
      }
      @Override
      public Adapter caseSignalArgumentVariable(SignalArgumentVariable object)
      {
        return createSignalArgumentVariableAdapter();
      }
      @Override
      public Adapter caseSignalArgumentExpression(SignalArgumentExpression object)
      {
        return createSignalArgumentExpressionAdapter();
      }
      @Override
      public Adapter caseAssignment(Assignment object)
      {
        return createAssignmentAdapter();
      }
      @Override
      public Adapter caseSendSignal(SendSignal object)
      {
        return createSendSignalAdapter();
      }
      @Override
      public Adapter caseObject(org.xtext.textualSlco.Object object)
      {
        return createObjectAdapter();
      }
      @Override
      public Adapter casePort(Port object)
      {
        return createPortAdapter();
      }
      @Override
      public Adapter caseChannel(Channel object)
      {
        return createChannelAdapter();
      }
      @Override
      public Adapter caseUnidirectionalChannel(UnidirectionalChannel object)
      {
        return createUnidirectionalChannelAdapter();
      }
      @Override
      public Adapter caseBidirectionalChannel(BidirectionalChannel object)
      {
        return createBidirectionalChannelAdapter();
      }
      @Override
      public Adapter caseArgumentType(ArgumentType object)
      {
        return createArgumentTypeAdapter();
      }
      @Override
      public Adapter caseVariable(Variable object)
      {
        return createVariableAdapter();
      }
      @Override
      public Adapter caseExpression(Expression object)
      {
        return createExpressionAdapter();
      }
      @Override
      public Adapter caseBooleanConstantExpression(BooleanConstantExpression object)
      {
        return createBooleanConstantExpressionAdapter();
      }
      @Override
      public Adapter caseIntegerConstantExpression(IntegerConstantExpression object)
      {
        return createIntegerConstantExpressionAdapter();
      }
      @Override
      public Adapter caseStringConstantExpression(StringConstantExpression object)
      {
        return createStringConstantExpressionAdapter();
      }
      @Override
      public Adapter caseVariableReference(VariableReference object)
      {
        return createVariableReferenceAdapter();
      }
      @Override
      public Adapter caseVariableExpression(VariableExpression object)
      {
        return createVariableExpressionAdapter();
      }
      @Override
      public Adapter caseBinaryOperatorExpression(BinaryOperatorExpression object)
      {
        return createBinaryOperatorExpressionAdapter();
      }
      @Override
      public Adapter defaultCase(EObject object)
      {
        return createEObjectAdapter();
      }
    };

  /**
   * Creates an adapter for the <code>target</code>.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param target the object to adapt.
   * @return the adapter for the <code>target</code>.
   * @generated
   */
  @Override
  public Adapter createAdapter(Notifier target)
  {
    return modelSwitch.doSwitch((EObject)target);
  }


  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.Model <em>Model</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.Model
   * @generated
   */
  public Adapter createModelAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.Class <em>Class</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.Class
   * @generated
   */
  public Adapter createClassAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.StateMachine <em>State Machine</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.StateMachine
   * @generated
   */
  public Adapter createStateMachineAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.Vertex <em>Vertex</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.Vertex
   * @generated
   */
  public Adapter createVertexAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.Initial <em>Initial</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.Initial
   * @generated
   */
  public Adapter createInitialAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.State <em>State</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.State
   * @generated
   */
  public Adapter createStateAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.Final <em>Final</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.Final
   * @generated
   */
  public Adapter createFinalAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.VertexReference <em>Vertex Reference</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.VertexReference
   * @generated
   */
  public Adapter createVertexReferenceAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.Transition <em>Transition</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.Transition
   * @generated
   */
  public Adapter createTransitionAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.Statement <em>Statement</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.Statement
   * @generated
   */
  public Adapter createStatementAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.Delay <em>Delay</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.Delay
   * @generated
   */
  public Adapter createDelayAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.PortReference <em>Port Reference</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.PortReference
   * @generated
   */
  public Adapter createPortReferenceAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.SignalReception <em>Signal Reception</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.SignalReception
   * @generated
   */
  public Adapter createSignalReceptionAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.SignalArgument <em>Signal Argument</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.SignalArgument
   * @generated
   */
  public Adapter createSignalArgumentAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.SignalArgumentVariable <em>Signal Argument Variable</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.SignalArgumentVariable
   * @generated
   */
  public Adapter createSignalArgumentVariableAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.SignalArgumentExpression <em>Signal Argument Expression</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.SignalArgumentExpression
   * @generated
   */
  public Adapter createSignalArgumentExpressionAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.Assignment <em>Assignment</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.Assignment
   * @generated
   */
  public Adapter createAssignmentAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.SendSignal <em>Send Signal</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.SendSignal
   * @generated
   */
  public Adapter createSendSignalAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.Object <em>Object</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.Object
   * @generated
   */
  public Adapter createObjectAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.Port <em>Port</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.Port
   * @generated
   */
  public Adapter createPortAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.Channel <em>Channel</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.Channel
   * @generated
   */
  public Adapter createChannelAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.UnidirectionalChannel <em>Unidirectional Channel</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.UnidirectionalChannel
   * @generated
   */
  public Adapter createUnidirectionalChannelAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.BidirectionalChannel <em>Bidirectional Channel</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.BidirectionalChannel
   * @generated
   */
  public Adapter createBidirectionalChannelAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.ArgumentType <em>Argument Type</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.ArgumentType
   * @generated
   */
  public Adapter createArgumentTypeAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.Variable <em>Variable</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.Variable
   * @generated
   */
  public Adapter createVariableAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.Expression <em>Expression</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.Expression
   * @generated
   */
  public Adapter createExpressionAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.BooleanConstantExpression <em>Boolean Constant Expression</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.BooleanConstantExpression
   * @generated
   */
  public Adapter createBooleanConstantExpressionAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.IntegerConstantExpression <em>Integer Constant Expression</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.IntegerConstantExpression
   * @generated
   */
  public Adapter createIntegerConstantExpressionAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.StringConstantExpression <em>String Constant Expression</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.StringConstantExpression
   * @generated
   */
  public Adapter createStringConstantExpressionAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.VariableReference <em>Variable Reference</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.VariableReference
   * @generated
   */
  public Adapter createVariableReferenceAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.VariableExpression <em>Variable Expression</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.VariableExpression
   * @generated
   */
  public Adapter createVariableExpressionAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for an object of class '{@link org.xtext.textualSlco.BinaryOperatorExpression <em>Binary Operator Expression</em>}'.
   * <!-- begin-user-doc -->
   * This default implementation returns null so that we can easily ignore cases;
   * it's useful to ignore a case when inheritance will catch all the cases anyway.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @see org.xtext.textualSlco.BinaryOperatorExpression
   * @generated
   */
  public Adapter createBinaryOperatorExpressionAdapter()
  {
    return null;
  }

  /**
   * Creates a new adapter for the default case.
   * <!-- begin-user-doc -->
   * This default implementation returns null.
   * <!-- end-user-doc -->
   * @return the new adapter.
   * @generated
   */
  public Adapter createEObjectAdapter()
  {
    return null;
  }

} //TextualSlcoAdapterFactory
