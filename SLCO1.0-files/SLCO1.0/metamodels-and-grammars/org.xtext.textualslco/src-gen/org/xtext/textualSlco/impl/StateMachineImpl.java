/**
 */
package org.xtext.textualSlco.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import org.xtext.textualSlco.StateMachine;
import org.xtext.textualSlco.TextualSlcoPackage;
import org.xtext.textualSlco.Transition;
import org.xtext.textualSlco.Variable;
import org.xtext.textualSlco.Vertex;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>State Machine</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.impl.StateMachineImpl#getName <em>Name</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.StateMachineImpl#getVariables <em>Variables</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.StateMachineImpl#getVertices <em>Vertices</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.StateMachineImpl#getTransitions <em>Transitions</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class StateMachineImpl extends MinimalEObjectImpl.Container implements StateMachine
{
  /**
   * The default value of the '{@link #getName() <em>Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getName()
   * @generated
   * @ordered
   */
  protected static final String NAME_EDEFAULT = null;

  /**
   * The cached value of the '{@link #getName() <em>Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getName()
   * @generated
   * @ordered
   */
  protected String name = NAME_EDEFAULT;

  /**
   * The cached value of the '{@link #getVariables() <em>Variables</em>}' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getVariables()
   * @generated
   * @ordered
   */
  protected EList<Variable> variables;

  /**
   * The cached value of the '{@link #getVertices() <em>Vertices</em>}' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getVertices()
   * @generated
   * @ordered
   */
  protected EList<Vertex> vertices;

  /**
   * The cached value of the '{@link #getTransitions() <em>Transitions</em>}' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getTransitions()
   * @generated
   * @ordered
   */
  protected EList<Transition> transitions;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected StateMachineImpl()
  {
    super();
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  protected EClass eStaticClass()
  {
    return TextualSlcoPackage.Literals.STATE_MACHINE;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public String getName()
  {
    return name;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void setName(String newName)
  {
    String oldName = name;
    name = newName;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.STATE_MACHINE__NAME, oldName, name));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EList<Variable> getVariables()
  {
    if (variables == null)
    {
      variables = new EObjectContainmentEList<Variable>(Variable.class, this, TextualSlcoPackage.STATE_MACHINE__VARIABLES);
    }
    return variables;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EList<Vertex> getVertices()
  {
    if (vertices == null)
    {
      vertices = new EObjectContainmentEList<Vertex>(Vertex.class, this, TextualSlcoPackage.STATE_MACHINE__VERTICES);
    }
    return vertices;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EList<Transition> getTransitions()
  {
    if (transitions == null)
    {
      transitions = new EObjectContainmentEList<Transition>(Transition.class, this, TextualSlcoPackage.STATE_MACHINE__TRANSITIONS);
    }
    return transitions;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs)
  {
    switch (featureID)
    {
      case TextualSlcoPackage.STATE_MACHINE__VARIABLES:
        return ((InternalEList<?>)getVariables()).basicRemove(otherEnd, msgs);
      case TextualSlcoPackage.STATE_MACHINE__VERTICES:
        return ((InternalEList<?>)getVertices()).basicRemove(otherEnd, msgs);
      case TextualSlcoPackage.STATE_MACHINE__TRANSITIONS:
        return ((InternalEList<?>)getTransitions()).basicRemove(otherEnd, msgs);
    }
    return super.eInverseRemove(otherEnd, featureID, msgs);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public Object eGet(int featureID, boolean resolve, boolean coreType)
  {
    switch (featureID)
    {
      case TextualSlcoPackage.STATE_MACHINE__NAME:
        return getName();
      case TextualSlcoPackage.STATE_MACHINE__VARIABLES:
        return getVariables();
      case TextualSlcoPackage.STATE_MACHINE__VERTICES:
        return getVertices();
      case TextualSlcoPackage.STATE_MACHINE__TRANSITIONS:
        return getTransitions();
    }
    return super.eGet(featureID, resolve, coreType);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @SuppressWarnings("unchecked")
  @Override
  public void eSet(int featureID, Object newValue)
  {
    switch (featureID)
    {
      case TextualSlcoPackage.STATE_MACHINE__NAME:
        setName((String)newValue);
        return;
      case TextualSlcoPackage.STATE_MACHINE__VARIABLES:
        getVariables().clear();
        getVariables().addAll((Collection<? extends Variable>)newValue);
        return;
      case TextualSlcoPackage.STATE_MACHINE__VERTICES:
        getVertices().clear();
        getVertices().addAll((Collection<? extends Vertex>)newValue);
        return;
      case TextualSlcoPackage.STATE_MACHINE__TRANSITIONS:
        getTransitions().clear();
        getTransitions().addAll((Collection<? extends Transition>)newValue);
        return;
    }
    super.eSet(featureID, newValue);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void eUnset(int featureID)
  {
    switch (featureID)
    {
      case TextualSlcoPackage.STATE_MACHINE__NAME:
        setName(NAME_EDEFAULT);
        return;
      case TextualSlcoPackage.STATE_MACHINE__VARIABLES:
        getVariables().clear();
        return;
      case TextualSlcoPackage.STATE_MACHINE__VERTICES:
        getVertices().clear();
        return;
      case TextualSlcoPackage.STATE_MACHINE__TRANSITIONS:
        getTransitions().clear();
        return;
    }
    super.eUnset(featureID);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public boolean eIsSet(int featureID)
  {
    switch (featureID)
    {
      case TextualSlcoPackage.STATE_MACHINE__NAME:
        return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
      case TextualSlcoPackage.STATE_MACHINE__VARIABLES:
        return variables != null && !variables.isEmpty();
      case TextualSlcoPackage.STATE_MACHINE__VERTICES:
        return vertices != null && !vertices.isEmpty();
      case TextualSlcoPackage.STATE_MACHINE__TRANSITIONS:
        return transitions != null && !transitions.isEmpty();
    }
    return super.eIsSet(featureID);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public String toString()
  {
    if (eIsProxy()) return super.toString();

    StringBuffer result = new StringBuffer(super.toString());
    result.append(" (name: ");
    result.append(name);
    result.append(')');
    return result.toString();
  }

} //StateMachineImpl
