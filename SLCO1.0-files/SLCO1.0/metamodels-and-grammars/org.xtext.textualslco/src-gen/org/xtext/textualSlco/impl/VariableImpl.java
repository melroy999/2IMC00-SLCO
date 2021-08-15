/**
 */
package org.xtext.textualSlco.impl;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

import org.xtext.textualSlco.Expression;
import org.xtext.textualSlco.PrimitiveType;
import org.xtext.textualSlco.TextualSlcoPackage;
import org.xtext.textualSlco.Variable;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Variable</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.impl.VariableImpl#getType <em>Type</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.VariableImpl#getName <em>Name</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.VariableImpl#getIntialValue <em>Intial Value</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class VariableImpl extends MinimalEObjectImpl.Container implements Variable
{
  /**
   * The default value of the '{@link #getType() <em>Type</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getType()
   * @generated
   * @ordered
   */
  protected static final PrimitiveType TYPE_EDEFAULT = PrimitiveType.INTEGER;

  /**
   * The cached value of the '{@link #getType() <em>Type</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getType()
   * @generated
   * @ordered
   */
  protected PrimitiveType type = TYPE_EDEFAULT;

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
   * The cached value of the '{@link #getIntialValue() <em>Intial Value</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getIntialValue()
   * @generated
   * @ordered
   */
  protected Expression intialValue;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected VariableImpl()
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
    return TextualSlcoPackage.Literals.VARIABLE;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public PrimitiveType getType()
  {
    return type;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void setType(PrimitiveType newType)
  {
    PrimitiveType oldType = type;
    type = newType == null ? TYPE_EDEFAULT : newType;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.VARIABLE__TYPE, oldType, type));
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
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.VARIABLE__NAME, oldName, name));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public Expression getIntialValue()
  {
    return intialValue;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public NotificationChain basicSetIntialValue(Expression newIntialValue, NotificationChain msgs)
  {
    Expression oldIntialValue = intialValue;
    intialValue = newIntialValue;
    if (eNotificationRequired())
    {
      ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.VARIABLE__INTIAL_VALUE, oldIntialValue, newIntialValue);
      if (msgs == null) msgs = notification; else msgs.add(notification);
    }
    return msgs;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void setIntialValue(Expression newIntialValue)
  {
    if (newIntialValue != intialValue)
    {
      NotificationChain msgs = null;
      if (intialValue != null)
        msgs = ((InternalEObject)intialValue).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - TextualSlcoPackage.VARIABLE__INTIAL_VALUE, null, msgs);
      if (newIntialValue != null)
        msgs = ((InternalEObject)newIntialValue).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - TextualSlcoPackage.VARIABLE__INTIAL_VALUE, null, msgs);
      msgs = basicSetIntialValue(newIntialValue, msgs);
      if (msgs != null) msgs.dispatch();
    }
    else if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.VARIABLE__INTIAL_VALUE, newIntialValue, newIntialValue));
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
      case TextualSlcoPackage.VARIABLE__INTIAL_VALUE:
        return basicSetIntialValue(null, msgs);
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
      case TextualSlcoPackage.VARIABLE__TYPE:
        return getType();
      case TextualSlcoPackage.VARIABLE__NAME:
        return getName();
      case TextualSlcoPackage.VARIABLE__INTIAL_VALUE:
        return getIntialValue();
    }
    return super.eGet(featureID, resolve, coreType);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void eSet(int featureID, Object newValue)
  {
    switch (featureID)
    {
      case TextualSlcoPackage.VARIABLE__TYPE:
        setType((PrimitiveType)newValue);
        return;
      case TextualSlcoPackage.VARIABLE__NAME:
        setName((String)newValue);
        return;
      case TextualSlcoPackage.VARIABLE__INTIAL_VALUE:
        setIntialValue((Expression)newValue);
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
      case TextualSlcoPackage.VARIABLE__TYPE:
        setType(TYPE_EDEFAULT);
        return;
      case TextualSlcoPackage.VARIABLE__NAME:
        setName(NAME_EDEFAULT);
        return;
      case TextualSlcoPackage.VARIABLE__INTIAL_VALUE:
        setIntialValue((Expression)null);
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
      case TextualSlcoPackage.VARIABLE__TYPE:
        return type != TYPE_EDEFAULT;
      case TextualSlcoPackage.VARIABLE__NAME:
        return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
      case TextualSlcoPackage.VARIABLE__INTIAL_VALUE:
        return intialValue != null;
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
    result.append(" (type: ");
    result.append(type);
    result.append(", name: ");
    result.append(name);
    result.append(')');
    return result.toString();
  }

} //VariableImpl
