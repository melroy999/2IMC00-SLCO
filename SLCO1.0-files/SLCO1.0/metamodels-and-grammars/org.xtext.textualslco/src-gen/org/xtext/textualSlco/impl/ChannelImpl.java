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

import org.xtext.textualSlco.ArgumentType;
import org.xtext.textualSlco.Channel;
import org.xtext.textualSlco.ChannelType;
import org.xtext.textualSlco.TextualSlcoPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Channel</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.impl.ChannelImpl#getName <em>Name</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.ChannelImpl#getArgumentTypes <em>Argument Types</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.ChannelImpl#getChannelType <em>Channel Type</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ChannelImpl extends MinimalEObjectImpl.Container implements Channel
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
   * The cached value of the '{@link #getArgumentTypes() <em>Argument Types</em>}' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getArgumentTypes()
   * @generated
   * @ordered
   */
  protected EList<ArgumentType> argumentTypes;

  /**
   * The default value of the '{@link #getChannelType() <em>Channel Type</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getChannelType()
   * @generated
   * @ordered
   */
  protected static final ChannelType CHANNEL_TYPE_EDEFAULT = ChannelType.SYNCHRONOUS;

  /**
   * The cached value of the '{@link #getChannelType() <em>Channel Type</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getChannelType()
   * @generated
   * @ordered
   */
  protected ChannelType channelType = CHANNEL_TYPE_EDEFAULT;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected ChannelImpl()
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
    return TextualSlcoPackage.Literals.CHANNEL;
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
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.CHANNEL__NAME, oldName, name));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EList<ArgumentType> getArgumentTypes()
  {
    if (argumentTypes == null)
    {
      argumentTypes = new EObjectContainmentEList<ArgumentType>(ArgumentType.class, this, TextualSlcoPackage.CHANNEL__ARGUMENT_TYPES);
    }
    return argumentTypes;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public ChannelType getChannelType()
  {
    return channelType;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void setChannelType(ChannelType newChannelType)
  {
    ChannelType oldChannelType = channelType;
    channelType = newChannelType == null ? CHANNEL_TYPE_EDEFAULT : newChannelType;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.CHANNEL__CHANNEL_TYPE, oldChannelType, channelType));
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
      case TextualSlcoPackage.CHANNEL__ARGUMENT_TYPES:
        return ((InternalEList<?>)getArgumentTypes()).basicRemove(otherEnd, msgs);
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
      case TextualSlcoPackage.CHANNEL__NAME:
        return getName();
      case TextualSlcoPackage.CHANNEL__ARGUMENT_TYPES:
        return getArgumentTypes();
      case TextualSlcoPackage.CHANNEL__CHANNEL_TYPE:
        return getChannelType();
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
      case TextualSlcoPackage.CHANNEL__NAME:
        setName((String)newValue);
        return;
      case TextualSlcoPackage.CHANNEL__ARGUMENT_TYPES:
        getArgumentTypes().clear();
        getArgumentTypes().addAll((Collection<? extends ArgumentType>)newValue);
        return;
      case TextualSlcoPackage.CHANNEL__CHANNEL_TYPE:
        setChannelType((ChannelType)newValue);
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
      case TextualSlcoPackage.CHANNEL__NAME:
        setName(NAME_EDEFAULT);
        return;
      case TextualSlcoPackage.CHANNEL__ARGUMENT_TYPES:
        getArgumentTypes().clear();
        return;
      case TextualSlcoPackage.CHANNEL__CHANNEL_TYPE:
        setChannelType(CHANNEL_TYPE_EDEFAULT);
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
      case TextualSlcoPackage.CHANNEL__NAME:
        return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
      case TextualSlcoPackage.CHANNEL__ARGUMENT_TYPES:
        return argumentTypes != null && !argumentTypes.isEmpty();
      case TextualSlcoPackage.CHANNEL__CHANNEL_TYPE:
        return channelType != CHANNEL_TYPE_EDEFAULT;
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
    result.append(", channelType: ");
    result.append(channelType);
    result.append(')');
    return result.toString();
  }

} //ChannelImpl
