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

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import org.xtext.textualSlco.Expression;
import org.xtext.textualSlco.PortReference;
import org.xtext.textualSlco.SendSignal;
import org.xtext.textualSlco.TextualSlcoPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Send Signal</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.impl.SendSignalImpl#getSignalName <em>Signal Name</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.SendSignalImpl#getArguments <em>Arguments</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.SendSignalImpl#getPort <em>Port</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class SendSignalImpl extends StatementImpl implements SendSignal
{
  /**
   * The default value of the '{@link #getSignalName() <em>Signal Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getSignalName()
   * @generated
   * @ordered
   */
  protected static final String SIGNAL_NAME_EDEFAULT = null;

  /**
   * The cached value of the '{@link #getSignalName() <em>Signal Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getSignalName()
   * @generated
   * @ordered
   */
  protected String signalName = SIGNAL_NAME_EDEFAULT;

  /**
   * The cached value of the '{@link #getArguments() <em>Arguments</em>}' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getArguments()
   * @generated
   * @ordered
   */
  protected EList<Expression> arguments;

  /**
   * The cached value of the '{@link #getPort() <em>Port</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getPort()
   * @generated
   * @ordered
   */
  protected PortReference port;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected SendSignalImpl()
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
    return TextualSlcoPackage.Literals.SEND_SIGNAL;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public String getSignalName()
  {
    return signalName;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void setSignalName(String newSignalName)
  {
    String oldSignalName = signalName;
    signalName = newSignalName;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.SEND_SIGNAL__SIGNAL_NAME, oldSignalName, signalName));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EList<Expression> getArguments()
  {
    if (arguments == null)
    {
      arguments = new EObjectContainmentEList<Expression>(Expression.class, this, TextualSlcoPackage.SEND_SIGNAL__ARGUMENTS);
    }
    return arguments;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public PortReference getPort()
  {
    return port;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public NotificationChain basicSetPort(PortReference newPort, NotificationChain msgs)
  {
    PortReference oldPort = port;
    port = newPort;
    if (eNotificationRequired())
    {
      ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.SEND_SIGNAL__PORT, oldPort, newPort);
      if (msgs == null) msgs = notification; else msgs.add(notification);
    }
    return msgs;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void setPort(PortReference newPort)
  {
    if (newPort != port)
    {
      NotificationChain msgs = null;
      if (port != null)
        msgs = ((InternalEObject)port).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - TextualSlcoPackage.SEND_SIGNAL__PORT, null, msgs);
      if (newPort != null)
        msgs = ((InternalEObject)newPort).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - TextualSlcoPackage.SEND_SIGNAL__PORT, null, msgs);
      msgs = basicSetPort(newPort, msgs);
      if (msgs != null) msgs.dispatch();
    }
    else if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.SEND_SIGNAL__PORT, newPort, newPort));
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
      case TextualSlcoPackage.SEND_SIGNAL__ARGUMENTS:
        return ((InternalEList<?>)getArguments()).basicRemove(otherEnd, msgs);
      case TextualSlcoPackage.SEND_SIGNAL__PORT:
        return basicSetPort(null, msgs);
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
      case TextualSlcoPackage.SEND_SIGNAL__SIGNAL_NAME:
        return getSignalName();
      case TextualSlcoPackage.SEND_SIGNAL__ARGUMENTS:
        return getArguments();
      case TextualSlcoPackage.SEND_SIGNAL__PORT:
        return getPort();
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
      case TextualSlcoPackage.SEND_SIGNAL__SIGNAL_NAME:
        setSignalName((String)newValue);
        return;
      case TextualSlcoPackage.SEND_SIGNAL__ARGUMENTS:
        getArguments().clear();
        getArguments().addAll((Collection<? extends Expression>)newValue);
        return;
      case TextualSlcoPackage.SEND_SIGNAL__PORT:
        setPort((PortReference)newValue);
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
      case TextualSlcoPackage.SEND_SIGNAL__SIGNAL_NAME:
        setSignalName(SIGNAL_NAME_EDEFAULT);
        return;
      case TextualSlcoPackage.SEND_SIGNAL__ARGUMENTS:
        getArguments().clear();
        return;
      case TextualSlcoPackage.SEND_SIGNAL__PORT:
        setPort((PortReference)null);
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
      case TextualSlcoPackage.SEND_SIGNAL__SIGNAL_NAME:
        return SIGNAL_NAME_EDEFAULT == null ? signalName != null : !SIGNAL_NAME_EDEFAULT.equals(signalName);
      case TextualSlcoPackage.SEND_SIGNAL__ARGUMENTS:
        return arguments != null && !arguments.isEmpty();
      case TextualSlcoPackage.SEND_SIGNAL__PORT:
        return port != null;
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
    result.append(" (signalName: ");
    result.append(signalName);
    result.append(')');
    return result.toString();
  }

} //SendSignalImpl
