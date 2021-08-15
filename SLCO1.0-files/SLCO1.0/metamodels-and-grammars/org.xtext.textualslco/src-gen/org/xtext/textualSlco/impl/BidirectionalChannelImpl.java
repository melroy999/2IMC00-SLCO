/**
 */
package org.xtext.textualSlco.impl;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.xtext.textualSlco.BidirectionalChannel;
import org.xtext.textualSlco.PortReference;
import org.xtext.textualSlco.TextualSlcoPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Bidirectional Channel</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.impl.BidirectionalChannelImpl#getObject1 <em>Object1</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.BidirectionalChannelImpl#getPort1 <em>Port1</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.BidirectionalChannelImpl#getObject2 <em>Object2</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.BidirectionalChannelImpl#getPort2 <em>Port2</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class BidirectionalChannelImpl extends ChannelImpl implements BidirectionalChannel
{
  /**
   * The cached value of the '{@link #getObject1() <em>Object1</em>}' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getObject1()
   * @generated
   * @ordered
   */
  protected org.xtext.textualSlco.Object object1;

  /**
   * The cached value of the '{@link #getPort1() <em>Port1</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getPort1()
   * @generated
   * @ordered
   */
  protected PortReference port1;

  /**
   * The cached value of the '{@link #getObject2() <em>Object2</em>}' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getObject2()
   * @generated
   * @ordered
   */
  protected org.xtext.textualSlco.Object object2;

  /**
   * The cached value of the '{@link #getPort2() <em>Port2</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getPort2()
   * @generated
   * @ordered
   */
  protected PortReference port2;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected BidirectionalChannelImpl()
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
    return TextualSlcoPackage.Literals.BIDIRECTIONAL_CHANNEL;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public org.xtext.textualSlco.Object getObject1()
  {
    if (object1 != null && object1.eIsProxy())
    {
      InternalEObject oldObject1 = (InternalEObject)object1;
      object1 = (org.xtext.textualSlco.Object)eResolveProxy(oldObject1);
      if (object1 != oldObject1)
      {
        if (eNotificationRequired())
          eNotify(new ENotificationImpl(this, Notification.RESOLVE, TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT1, oldObject1, object1));
      }
    }
    return object1;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public org.xtext.textualSlco.Object basicGetObject1()
  {
    return object1;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void setObject1(org.xtext.textualSlco.Object newObject1)
  {
    org.xtext.textualSlco.Object oldObject1 = object1;
    object1 = newObject1;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT1, oldObject1, object1));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public PortReference getPort1()
  {
    return port1;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public NotificationChain basicSetPort1(PortReference newPort1, NotificationChain msgs)
  {
    PortReference oldPort1 = port1;
    port1 = newPort1;
    if (eNotificationRequired())
    {
      ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1, oldPort1, newPort1);
      if (msgs == null) msgs = notification; else msgs.add(notification);
    }
    return msgs;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void setPort1(PortReference newPort1)
  {
    if (newPort1 != port1)
    {
      NotificationChain msgs = null;
      if (port1 != null)
        msgs = ((InternalEObject)port1).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1, null, msgs);
      if (newPort1 != null)
        msgs = ((InternalEObject)newPort1).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1, null, msgs);
      msgs = basicSetPort1(newPort1, msgs);
      if (msgs != null) msgs.dispatch();
    }
    else if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1, newPort1, newPort1));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public org.xtext.textualSlco.Object getObject2()
  {
    if (object2 != null && object2.eIsProxy())
    {
      InternalEObject oldObject2 = (InternalEObject)object2;
      object2 = (org.xtext.textualSlco.Object)eResolveProxy(oldObject2);
      if (object2 != oldObject2)
      {
        if (eNotificationRequired())
          eNotify(new ENotificationImpl(this, Notification.RESOLVE, TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT2, oldObject2, object2));
      }
    }
    return object2;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public org.xtext.textualSlco.Object basicGetObject2()
  {
    return object2;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void setObject2(org.xtext.textualSlco.Object newObject2)
  {
    org.xtext.textualSlco.Object oldObject2 = object2;
    object2 = newObject2;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT2, oldObject2, object2));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public PortReference getPort2()
  {
    return port2;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public NotificationChain basicSetPort2(PortReference newPort2, NotificationChain msgs)
  {
    PortReference oldPort2 = port2;
    port2 = newPort2;
    if (eNotificationRequired())
    {
      ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2, oldPort2, newPort2);
      if (msgs == null) msgs = notification; else msgs.add(notification);
    }
    return msgs;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void setPort2(PortReference newPort2)
  {
    if (newPort2 != port2)
    {
      NotificationChain msgs = null;
      if (port2 != null)
        msgs = ((InternalEObject)port2).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2, null, msgs);
      if (newPort2 != null)
        msgs = ((InternalEObject)newPort2).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2, null, msgs);
      msgs = basicSetPort2(newPort2, msgs);
      if (msgs != null) msgs.dispatch();
    }
    else if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2, newPort2, newPort2));
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
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1:
        return basicSetPort1(null, msgs);
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2:
        return basicSetPort2(null, msgs);
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
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT1:
        if (resolve) return getObject1();
        return basicGetObject1();
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1:
        return getPort1();
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT2:
        if (resolve) return getObject2();
        return basicGetObject2();
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2:
        return getPort2();
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
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT1:
        setObject1((org.xtext.textualSlco.Object)newValue);
        return;
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1:
        setPort1((PortReference)newValue);
        return;
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT2:
        setObject2((org.xtext.textualSlco.Object)newValue);
        return;
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2:
        setPort2((PortReference)newValue);
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
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT1:
        setObject1((org.xtext.textualSlco.Object)null);
        return;
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1:
        setPort1((PortReference)null);
        return;
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT2:
        setObject2((org.xtext.textualSlco.Object)null);
        return;
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2:
        setPort2((PortReference)null);
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
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT1:
        return object1 != null;
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT1:
        return port1 != null;
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__OBJECT2:
        return object2 != null;
      case TextualSlcoPackage.BIDIRECTIONAL_CHANNEL__PORT2:
        return port2 != null;
    }
    return super.eIsSet(featureID);
  }

} //BidirectionalChannelImpl
