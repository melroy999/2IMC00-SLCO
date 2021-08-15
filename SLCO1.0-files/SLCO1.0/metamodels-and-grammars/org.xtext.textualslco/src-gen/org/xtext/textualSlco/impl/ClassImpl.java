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

import org.xtext.textualSlco.Port;
import org.xtext.textualSlco.StateMachine;
import org.xtext.textualSlco.TextualSlcoPackage;
import org.xtext.textualSlco.Variable;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Class</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.impl.ClassImpl#getName <em>Name</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.ClassImpl#getVariables <em>Variables</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.ClassImpl#getPorts <em>Ports</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.ClassImpl#getStateMachines <em>State Machines</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ClassImpl extends MinimalEObjectImpl.Container implements org.xtext.textualSlco.Class
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
   * The cached value of the '{@link #getPorts() <em>Ports</em>}' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getPorts()
   * @generated
   * @ordered
   */
  protected EList<Port> ports;

  /**
   * The cached value of the '{@link #getStateMachines() <em>State Machines</em>}' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getStateMachines()
   * @generated
   * @ordered
   */
  protected EList<StateMachine> stateMachines;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected ClassImpl()
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
    return TextualSlcoPackage.Literals.CLASS;
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
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.CLASS__NAME, oldName, name));
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
      variables = new EObjectContainmentEList<Variable>(Variable.class, this, TextualSlcoPackage.CLASS__VARIABLES);
    }
    return variables;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EList<Port> getPorts()
  {
    if (ports == null)
    {
      ports = new EObjectContainmentEList<Port>(Port.class, this, TextualSlcoPackage.CLASS__PORTS);
    }
    return ports;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EList<StateMachine> getStateMachines()
  {
    if (stateMachines == null)
    {
      stateMachines = new EObjectContainmentEList<StateMachine>(StateMachine.class, this, TextualSlcoPackage.CLASS__STATE_MACHINES);
    }
    return stateMachines;
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
      case TextualSlcoPackage.CLASS__VARIABLES:
        return ((InternalEList<?>)getVariables()).basicRemove(otherEnd, msgs);
      case TextualSlcoPackage.CLASS__PORTS:
        return ((InternalEList<?>)getPorts()).basicRemove(otherEnd, msgs);
      case TextualSlcoPackage.CLASS__STATE_MACHINES:
        return ((InternalEList<?>)getStateMachines()).basicRemove(otherEnd, msgs);
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
      case TextualSlcoPackage.CLASS__NAME:
        return getName();
      case TextualSlcoPackage.CLASS__VARIABLES:
        return getVariables();
      case TextualSlcoPackage.CLASS__PORTS:
        return getPorts();
      case TextualSlcoPackage.CLASS__STATE_MACHINES:
        return getStateMachines();
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
      case TextualSlcoPackage.CLASS__NAME:
        setName((String)newValue);
        return;
      case TextualSlcoPackage.CLASS__VARIABLES:
        getVariables().clear();
        getVariables().addAll((Collection<? extends Variable>)newValue);
        return;
      case TextualSlcoPackage.CLASS__PORTS:
        getPorts().clear();
        getPorts().addAll((Collection<? extends Port>)newValue);
        return;
      case TextualSlcoPackage.CLASS__STATE_MACHINES:
        getStateMachines().clear();
        getStateMachines().addAll((Collection<? extends StateMachine>)newValue);
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
      case TextualSlcoPackage.CLASS__NAME:
        setName(NAME_EDEFAULT);
        return;
      case TextualSlcoPackage.CLASS__VARIABLES:
        getVariables().clear();
        return;
      case TextualSlcoPackage.CLASS__PORTS:
        getPorts().clear();
        return;
      case TextualSlcoPackage.CLASS__STATE_MACHINES:
        getStateMachines().clear();
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
      case TextualSlcoPackage.CLASS__NAME:
        return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
      case TextualSlcoPackage.CLASS__VARIABLES:
        return variables != null && !variables.isEmpty();
      case TextualSlcoPackage.CLASS__PORTS:
        return ports != null && !ports.isEmpty();
      case TextualSlcoPackage.CLASS__STATE_MACHINES:
        return stateMachines != null && !stateMachines.isEmpty();
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

} //ClassImpl
