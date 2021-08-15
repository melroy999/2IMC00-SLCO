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

import org.xtext.textualSlco.Channel;
import org.xtext.textualSlco.Model;
import org.xtext.textualSlco.TextualSlcoPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Model</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.impl.ModelImpl#getName <em>Name</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.ModelImpl#getClasses <em>Classes</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.ModelImpl#getObjects <em>Objects</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.ModelImpl#getChannels <em>Channels</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class ModelImpl extends MinimalEObjectImpl.Container implements Model
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
   * The cached value of the '{@link #getClasses() <em>Classes</em>}' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getClasses()
   * @generated
   * @ordered
   */
  protected EList<org.xtext.textualSlco.Class> classes;

  /**
   * The cached value of the '{@link #getObjects() <em>Objects</em>}' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getObjects()
   * @generated
   * @ordered
   */
  protected EList<org.xtext.textualSlco.Object> objects;

  /**
   * The cached value of the '{@link #getChannels() <em>Channels</em>}' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getChannels()
   * @generated
   * @ordered
   */
  protected EList<Channel> channels;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected ModelImpl()
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
    return TextualSlcoPackage.Literals.MODEL;
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
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.MODEL__NAME, oldName, name));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EList<org.xtext.textualSlco.Class> getClasses()
  {
    if (classes == null)
    {
      classes = new EObjectContainmentEList<org.xtext.textualSlco.Class>(org.xtext.textualSlco.Class.class, this, TextualSlcoPackage.MODEL__CLASSES);
    }
    return classes;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EList<org.xtext.textualSlco.Object> getObjects()
  {
    if (objects == null)
    {
      objects = new EObjectContainmentEList<org.xtext.textualSlco.Object>(org.xtext.textualSlco.Object.class, this, TextualSlcoPackage.MODEL__OBJECTS);
    }
    return objects;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EList<Channel> getChannels()
  {
    if (channels == null)
    {
      channels = new EObjectContainmentEList<Channel>(Channel.class, this, TextualSlcoPackage.MODEL__CHANNELS);
    }
    return channels;
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
      case TextualSlcoPackage.MODEL__CLASSES:
        return ((InternalEList<?>)getClasses()).basicRemove(otherEnd, msgs);
      case TextualSlcoPackage.MODEL__OBJECTS:
        return ((InternalEList<?>)getObjects()).basicRemove(otherEnd, msgs);
      case TextualSlcoPackage.MODEL__CHANNELS:
        return ((InternalEList<?>)getChannels()).basicRemove(otherEnd, msgs);
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
      case TextualSlcoPackage.MODEL__NAME:
        return getName();
      case TextualSlcoPackage.MODEL__CLASSES:
        return getClasses();
      case TextualSlcoPackage.MODEL__OBJECTS:
        return getObjects();
      case TextualSlcoPackage.MODEL__CHANNELS:
        return getChannels();
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
      case TextualSlcoPackage.MODEL__NAME:
        setName((String)newValue);
        return;
      case TextualSlcoPackage.MODEL__CLASSES:
        getClasses().clear();
        getClasses().addAll((Collection<? extends org.xtext.textualSlco.Class>)newValue);
        return;
      case TextualSlcoPackage.MODEL__OBJECTS:
        getObjects().clear();
        getObjects().addAll((Collection<? extends org.xtext.textualSlco.Object>)newValue);
        return;
      case TextualSlcoPackage.MODEL__CHANNELS:
        getChannels().clear();
        getChannels().addAll((Collection<? extends Channel>)newValue);
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
      case TextualSlcoPackage.MODEL__NAME:
        setName(NAME_EDEFAULT);
        return;
      case TextualSlcoPackage.MODEL__CLASSES:
        getClasses().clear();
        return;
      case TextualSlcoPackage.MODEL__OBJECTS:
        getObjects().clear();
        return;
      case TextualSlcoPackage.MODEL__CHANNELS:
        getChannels().clear();
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
      case TextualSlcoPackage.MODEL__NAME:
        return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
      case TextualSlcoPackage.MODEL__CLASSES:
        return classes != null && !classes.isEmpty();
      case TextualSlcoPackage.MODEL__OBJECTS:
        return objects != null && !objects.isEmpty();
      case TextualSlcoPackage.MODEL__CHANNELS:
        return channels != null && !channels.isEmpty();
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

} //ModelImpl
